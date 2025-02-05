import fcntl
import logging
import os
import shutil
import signal
import socket
import time
import traceback
from contextlib import contextmanager
from functools import wraps
from threading import Thread

from internlm.accelerator.abstract_accelerator import get_accelerator
from internlm.core.context import global_context as gpc
from internlm.monitor.alert import send_feishu_msg_with_webhook
from internlm.utils.common import SingletonMeta

from .utils import get_job_key, set_env_var

logger = logging.getLogger(__file__)
internlm_accelerator = get_accelerator()


def internevo_monitor(feishu_alert=True, clean_run=True):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if feishu_alert:
                with initialize_monitor_manager(
                    job_name=gpc.config.JOB_NAME, alert_address=gpc.config.monitor.alert.feishu_alert_address
                ):
                    return execute_with_exception_handling(func, *args, **kwargs)
            else:
                return execute_with_exception_handling(func, *args, **kwargs)

        def execute_with_exception_handling(func, *args, **kwargs):
            if not clean_run:
                return func(*args, **kwargs)
            try:
                return func(*args, **kwargs)
            except Exception:
                hostname = socket.gethostname()
                logger.error(
                    f"Raise exception from {hostname} with rank id: {gpc.get_global_rank()}\n{traceback.format_exc()}",
                )
            finally:
                devices_per_node = internlm_accelerator.device_count()
                local_rank = gpc.get_global_rank() % devices_per_node
                if gpc.config.data.use_shm and local_rank == 0:
                    if os.path.exists(gpc.config.data.shm_path):
                        shutil.rmtree(gpc.config.data.shm_path)

        return wrapper

    return decorator


def send_alert_message(address: str = None, title: str = None, message: str = None):
    """
    Send alert messages to the given Feishu webhook address in log rank.

    Args:
        address (str): The alert address to be used to send message, defaults to None.
        title (str): The message title, defaults to None.
        message (str): The message body, defaults to None.
    """

    # send alert message only if the alert is enable
    if address is not None and gpc.config.monitor.alert.get("enable_feishu_alert", False) and gpc.is_rank_for_log():
        send_feishu_msg_with_webhook(
            webhook=address,
            title=title if title else get_job_key(),
            message=message,
        )


class MonitorTracker(Thread):
    """
    Track job status and alert to Feishu during job training.

    Args:
        alert_address (str): The Feishu webhook address for sending alerting messages.
        check_interval (float): The interval in seconds for monitoring checks. Defaults to 300.
        loss_spike_limit (float): The threshold for detecting loss value spikes. Defaults to 1.5.
    """

    def __init__(
        self,
        alert_address: str,
        check_interval: float = 300,
        loss_spike_limit: float = 1.5,
    ):
        super().__init__()
        self.alert_address = alert_address
        self.check_interval = check_interval
        self.loss_spike_limit = loss_spike_limit
        self.last_active_time = -1
        self.last_loss_value = -1
        self.stopped = False
        self.start()

    def run(self):
        """
        start the monitor tracker.
        """

        while not self.stopped:
            try:
                self._check_stuck()
                self._check_loss_spike()
            except Exception:
                continue
            time.sleep(self.check_interval)

    def _check_stuck(self):
        """
        Check training status for potential stuck condition.
        """

        new_active_time = -1
        if os.getenv("LAST_ACTIVE_TIMESTAMP") is not None:
            new_active_time = os.getenv("LAST_ACTIVE_TIMESTAMP")
        if int(new_active_time) <= int(self.last_active_time) and new_active_time != -1:
            self._send_alert("Training may be in stuck status, please check it.")
        self.last_active_time = new_active_time

    def _check_loss_spike(self):
        """
        Check for loss value spikes.
        """

        if gpc.is_rank_for_log():
            new_loss_value = -1
            new_step_id = -1
            if os.getenv("LOSS") is not None:
                new_loss_value = os.getenv("LOSS")
            if os.getenv("STEP_ID") is not None:
                new_step_id = os.getenv("STEP_ID")

            if (float(new_loss_value) / float(self.last_loss_value)) > self.loss_spike_limit and new_loss_value != -1:
                assert int(new_step_id) >= 0
                self._send_alert(
                    f"Checking periodically: Loss spike may be happened in step {new_step_id}, "
                    f"loss value from {self.last_loss_value} to {new_loss_value}, please check it."
                )

            self.last_loss_value = new_loss_value

    def _send_alert(self, message):
        """
        Send alerting message to the Feishu webhook address.

        Args:
            message (str): The alerting message to be sent.
        """

        send_alert_message(
            address=self.alert_address,
            message=message,
        )

    def stop(self):
        """
        Stop the monitor tracker.
        """

        self.stopped = True


class MonitorManager(metaclass=SingletonMeta):
    """
    Monitor Manager for managing monitor thread and monitoring training status.
    """

    def __init__(self, loss_spike_limit: float = 1.5) -> None:
        self.monitor_thread = None
        self.loss_spike_limit = loss_spike_limit
        self.last_step_loss = -1
        self.alert_file_path = None
        self.enable_alert = False
        self.light_monitor_address = None

    def monitor_loss_spike(self, alert_address: str = None, step_count: int = 0, cur_step_loss: float = 0.0):
        """Check loss value, if loss spike occurs, send alert message to Feishu."""
        if self.enable_alert:
            set_env_var(key="LOSS", value=cur_step_loss)
            set_env_var(key="STEP_ID", value=step_count)

            if self.last_step_loss != -1 and cur_step_loss > self.loss_spike_limit * self.last_step_loss:
                send_alert_message(
                    address=alert_address,
                    message=(
                        f"Checking step by step: Loss spike may be happened in step {step_count}, "
                        f"loss value from {self.last_step_loss} to {cur_step_loss}, please check it."
                    ),
                )
            self.last_step_loss = cur_step_loss

    def exception_should_be_alert(self, msg: str, alert_address: str = None):
        try:
            with open(self.alert_file_path, "a+") as f:
                fcntl.flock(f, fcntl.LOCK_EX)

                f.seek(0)
                if msg in f.read():
                    fcntl.flock(f, fcntl.LOCK_UN)
                    return False

                f.write(msg)
                fcntl.flock(f, fcntl.LOCK_UN)
                return True
        except Exception as err:
            send_alert_message(
                address=alert_address,
                message=f"Failed to open ALERT file: {err}",
            )
            return True

    def monitor_exception(self, alert_address: str = None, excp_info: str = None):
        """Catch and format exception information, send alert message to Feishu."""
        if self.enable_alert:
            filtered_trace = excp_info.split("\n")[-10:]
            format_trace = ""
            for line in filtered_trace:
                format_trace += "\n" + line

            message = f"Catch Exception from {socket.gethostname()} with rank id {gpc.get_global_rank()}:{format_trace}"
            if self.alert_file_path:
                if self.exception_should_be_alert(format_trace, alert_address):
                    send_feishu_msg_with_webhook(
                        webhook=alert_address,
                        title=get_job_key(),
                        message=message,
                    )
            else:
                send_alert_message(alert_address, message)

    def handle_sigterm(self, alert_address: str = None):
        """Catch SIGTERM signal, and send alert message to Feishu."""

        def sigterm_handler(sys_signal, frame):
            if self.enable_alert:
                print("receive frame: ", frame)
                print("receive signal: ", sys_signal)
                message = f"Process received signal {signal} and exited."
                send_alert_message(
                    address=alert_address,
                    message=message,
                )

        signal.signal(signal.SIGTERM, sigterm_handler)

    def start_monitor(
        self,
        job_name: str,
        alert_address: str,
        monitor_interval_seconds: int = 300,
        loss_spike_limit: float = 1.5,
    ):
        """
        Initialize and start monitor thread for checking training job status, loss spike and so on.

        Args:
            job_name (str): The training job name.
            alert_address (str): The Feishu webhook address for sending alert messages.
            monitor_interval_seconds (int): The time of monitor interval in seconds, defaults to 300.
            loss_spike_limit (float): The limit multiple of current loss to previous loss value, which means loss spike
                may be occurs, defaults to 1.5.
        """

        # initialize some variables for monitoring
        set_env_var(key="JOB_NAME", value=job_name)
        self.enable_alert = gpc.config.monitor.alert.get("enable_feishu_alert", False)

        if self.enable_alert:
            self.light_monitor_address = gpc.config.monitor.alert.get("light_monitor_address", None)
            # initialize alert file
            self.alert_file_path = gpc.config.monitor.alert.get("alert_file_path")
            if self.alert_file_path and gpc.is_rank_for_log():
                alert_file_dir = os.path.dirname(self.alert_file_path)
                os.makedirs(alert_file_dir, exist_ok=True)
                if os.path.exists(self.alert_file_path):
                    os.remove(self.alert_file_path)

            # start a monitor thread, periodically check the training status
            self.monitor_thread = MonitorTracker(
                alert_address=alert_address,
                check_interval=monitor_interval_seconds,
                loss_spike_limit=loss_spike_limit,
            )

    def stop_monitor(self):
        """Stop the monitor and alert thread."""
        if self.monitor_thread is not None:
            self.monitor_thread.stop()


monitor_manager = MonitorManager()


@contextmanager
def initialize_monitor_manager(job_name: str = None, alert_address: str = None):
    """
    Initialize monitor manager for monitoring training lifetime and alerting exception info to Feishu.

    Args:
        job_name (str): The training job name.
        alert_address (str): The Feishu webhook address for sending alert messages.
    """

    if alert_address is not None:
        try:
            monitor_manager.start_monitor(job_name=job_name, alert_address=alert_address)
            monitor_manager.handle_sigterm(alert_address=alert_address)
            send_alert_message(address=alert_address, message=f"Training in {socket.gethostname()} is starting.")
            yield
        finally:
            send_alert_message(address=alert_address, message=f"Training in {socket.gethostname()} completed.")
            monitor_manager.stop_monitor()
    else:
        yield
