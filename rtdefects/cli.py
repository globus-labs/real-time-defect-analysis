from threading import Thread
from argparse import ArgumentParser
from typing import Optional, List, Union
from pathlib import Path
from queue import Queue, Empty
from time import perf_counter, sleep
import logging
import json
import re

import cv2
import numpy as np
from funcx import FuncXClient
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileCreatedEvent, DirCreatedEvent


from rtdefects.flask import app


logger = logging.getLogger(__name__)
_config_path = Path(__file__).parent.joinpath('config.json')


def _funcx_func(data: bytes):
    """Function used for FuncX deployment of inference

    Inputs:
        data: TIF image data as a bytestring
    Returns:
        - A boolean array of the segmented regions
        - A dictionary of image analysis results
    """
    # Imports must be local to the function
    from rtdefects.function import perform_segmentation, analyze_defects
    from tempfile import NamedTemporaryFile
    from pathlib import Path
    import numpy as np
    import cv2

    # Load the file via disk
    temp = NamedTemporaryFile(suffix='.tif', mode='w+b', delete=False)
    try:
        temp.write(data)
        temp.close()

        image = cv2.imread(temp.name)
    finally:
        # Delete the file once complete
        Path(temp.name).unlink(missing_ok=True)

    # Preprocess the image data
    image = np.array(image, dtype=np.float32) / 255
    image = np.expand_dims(image, axis=0)

    # Perform the segmentation
    segment = perform_segmentation(image)

    # Make it into a bool array
    mask = segment[0, :, :, 0] > 0.9

    # Generate the analysis results
    defect_results = analyze_defects(mask)
    return mask, defect_results


def _set_config(function_id: Optional[str] = None, endpoint_id: Optional[str] = None):
    """Set the system configuration given the parser
    """

    # Read in the current configuration
    if _config_path.is_file():
        logger.info(f'Loading previous settings from {_config_path}')
        with open(_config_path, 'r') as fp:
            config = json.load(fp)
    else:
        config = {}

    # Define the configuration settings
    if function_id is not None:
        config['function_id'] = function_id
    if endpoint_id is not None:
        config['endpoint_id'] = endpoint_id

    # Save it
    with open(_config_path, 'w') as fp:
        json.dump(config, fp, indent=2)
    logger.info(f'Wrote configuration to {_config_path}')


def _register_function():
    """Register the inference function with FuncX"""

    client = FuncXClient()

    function_id = client.register_function(_funcx_func)
    _set_config(function_id=function_id)


class FuncXSubmitEventHandler(FileSystemEventHandler):
    """Submit a image processing task to FuncX when an image file is created"""

    def __init__(self, client: FuncXClient, func_id: str, endp_id: str, queue: Queue, file_regex: Optional[str] = None):
        """
        Args:
             client: FuncX client
             func_id: ID of the image processing function
             endp_id: Endpoint ID for execution
             queue: Queue to push results to
             file_regex: Regex string to match file formats
        """
        super().__init__()
        self.client = client
        self.func_id = func_id
        self.endp_id = endp_id
        self.queue = queue
        self.index = 0
        self.file_regex = re.compile(file_regex) if file_regex is not None else None

    def on_created(self, event: Union[FileCreatedEvent, DirCreatedEvent]):
        # Ignore directories
        if event.is_directory:
            logger.info('Created object is a directory. Skipping')
            return

        # Attempt to run it
        try:
            self._run_command(event)
        except BaseException as e:
            logger.warning(f'Submission for {event.src_path} failed. Error: {e}')

    def _run_command(self, event: FileCreatedEvent):
        """Read an image and send image processing task to FuncX

        Args:
            event: Event describing a file creation
        """
        # Match the filename
        if self.file_regex is not None:
            file_path = Path(event.src_path)
            if self.file_regex.match(file_path.name) is None:
                logger.info(f'Filename "{file_path}" did not match regex. Skipping')

        # Performance information
        detect_time = perf_counter()

        # Load the image from disk
        sleep(1.)
        with open(event.src_path, 'rb') as fp:
            image_data = fp.read()
        logger.info(f'Read a {len(image_data) / 1024 ** 2:.1f} MB image from {event.src_path}')

        # Submit it to FuncX for evaluation
        task_id = self.client.run(image_data, function_id=self.func_id, endpoint_id=self.endp_id)
        logger.info(f'Submitted task to FuncX. Task ID: {task_id}')

        # Push the task ID and submit time to the queue for processing
        self.index += 1
        self.queue.put((task_id, detect_time, self.index, Path(event.src_path)))


def main(args: Optional[List[str]] = None):
    """Launch service that automatically processes images and displays results as a web service"""

    # Make the argument parser
    parser = ArgumentParser()
    subparsers = parser.add_subparsers(dest='command', help='Which mode to launch the server in', required=True)

    # Add in the configuration settings
    config_parser = subparsers.add_parser('config', help='Define the configuration for the server')
    config_parser.add_argument('--function-id', help='UUID of the function to be run')
    config_parser.add_argument('--funcx-endpoint', help='FuncX endpoint on which to run image processing')

    # Add in the launch setting
    start_parser = subparsers.add_parser('start', help='Launch the processing service')
    start_parser.add_argument('--regex', default=r'.*.tiff?$', help='Regex to match files')
    start_parser.add_argument('watch_dir', help='Which directory to watch for new files')

    # Add in the register setting
    subparsers.add_parser('register', help='(Re)-register the funcX function')

    # Parse the input arguments
    args = parser.parse_args(args)

    # Make the logger
    logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)

    # Handle the configuration
    if args.command == 'config':
        return _set_config(function_id=args.function_id, endpoint_id=args.funcx_endpoint)
    elif args.command == 'register':
        return _register_function()

    assert args.command == 'start', f'Internal Error: The command "{args.command}" is not yet supported. Contact Logan'

    # Prepare the event handler
    client = FuncXClient()
    client.max_request_size = 50 * 1024 ** 2
    with open(_config_path, 'r') as fp:
        config = json.load(fp)
    exec_queue = Queue()
    handler = FuncXSubmitEventHandler(client, config['function_id'], config['endpoint_id'], exec_queue,
                                      file_regex=args.regex)

    # Prepare the watch directory
    watch_dir = Path(args.watch_dir)
    mask_dir = watch_dir.joinpath('masks')
    mask_dir.mkdir(exist_ok=True)

    # Prepare the watcher
    obs = Observer()
    obs.schedule(handler, path=args.watch_dir, recursive=False)
    obs.start()
    while True:
        try:
            # Wait for a task to be added the queue
            #  Note: Queue.get is not interruptable in Windows, hence the waiting logic
            while True:
                task_id = detect_time = index = img_path = None
                try:
                    task_id, detect_time, index, img_path = exec_queue.get(timeout=5)
                except Empty:
                    logger.debug('Waiting for task to complete')
                    continue
                else:
                    break
            logger.info(f'Waiting for task request {task_id} to complete')

            # Wait it for it finish from FuncX
            while (task := client.get_task(task_id))['pending']:
                sleep(1)
            mask, defect_info = task.pop('result')
            if mask is None:
                logger.warning(f'Task failure: {task["exception"]}')
                break
            rtt = perf_counter() - detect_time
            logger.info(f'Result received for {index}/{handler.index}. RTT: {rtt:.2f}s. Backlog: {exec_queue.qsize()}')

            # Save the mask to disk
            out_name = mask_dir.joinpath(img_path.name)
            cv2.imwrite(str(out_name), np.array(mask, dtype=np.float32))
            logger.info(f'Wrote output file to: {out_name}')

            # Write out the image defect information
            defect_info['mask-path'] = str(out_name)
            defect_info['image-path'] = str(img_path)
            with mask_dir.joinpath('defect-details.json').open('a') as fp:
                print(json.dumps(defect_info), file=fp)
        except KeyboardInterrupt:
            logger.info('Detected an interrupt. Stopping system')
            break
        except BaseException:
            obs.stop()
            logger.warning('Unexpected failure!')
            raise

    # Shut down the file reader
    obs.stop()
    obs.join()
