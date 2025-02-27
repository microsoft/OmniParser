import threading
from queue import Queue
from concurrent.futures import ThreadPoolExecutor
from paddleocr import PaddleOCR

class PaddleOCRPool:
    """
    A thread-safe pool of PaddleOCR instances.
    PaddleOCR is not thread-safe, so we need a pool of instances.
    """

    def __init__(
        self,
        pool_size=16,
        lang="en",
        use_angle_cls=False,
        use_gpu=False,
        show_log=False,
        max_batch_size=1024,
        use_dilation=True,
        det_db_score_mode="slow",
        rec_batch_num=1024,
    ):
        """
        Initialize a pool of PaddleOCR instances.

        Args:
            pool_size: Number of PaddleOCR instances to create
            lang: Language for OCR
            use_angle_cls: Whether to use angle classification
            use_gpu: Whether to use GPU
            show_log: Whether to show log
            max_batch_size: Max batch size
            use_dilation: Whether to use dilation
            det_db_score_mode: Detection DB score mode
            rec_batch_num: Recognition batch number
        """
        self.pool_size = pool_size
        self.ocr_queue = Queue()
        self.executor = ThreadPoolExecutor(max_workers=pool_size)
        self.ocr_params = {
            "lang": lang,
            "use_angle_cls": use_angle_cls,
            "use_gpu": use_gpu,
            "show_log": show_log,
            "max_batch_size": max_batch_size,
            "use_dilation": use_dilation,
            "det_db_score_mode": det_db_score_mode,
            "rec_batch_num": rec_batch_num,
        }
        # Add metrics tracking
        self.access_lock = threading.Lock()
        self.usage_metrics = {
            "total_requests": 0,
            "max_concurrent": 0,
            "current_in_use": 0,
        }
        self._initialize_pool()

    def _initialize_pool(self):
        """Initialize pool of PaddleOCR instances"""
        for _ in range(self.pool_size):
            ocr = PaddleOCR(**self.ocr_params)
            self.ocr_queue.put(ocr)

    def get_ocr(self):
        """Get a PaddleOCR instance from the pool"""
        with self.access_lock:
            self.usage_metrics["total_requests"] += 1
            self.usage_metrics["current_in_use"] += 1
            self.usage_metrics["max_concurrent"] = max(
                self.usage_metrics["max_concurrent"], 
                self.usage_metrics["current_in_use"]
            )
        return self.ocr_queue.get()

    def return_ocr(self, ocr):
        """Return a PaddleOCR instance to the pool"""
        self.ocr_queue.put(ocr)
        with self.access_lock:
            self.usage_metrics["current_in_use"] -= 1

    def process_image(self, image_np, cls=False):
        """
        Process an image with PaddleOCR in a thread-safe manner.

        Args:
            image_np: Numpy array of the image
            cls: Whether to use text direction classification

        Returns:
            OCR results or empty list on error
        """
        ocr = None
        try:
            ocr = self.get_ocr()
            result = ocr.ocr(image_np, cls=cls)
            self.return_ocr(ocr)
            return result
        except Exception as e:
            print(f"PaddleOCR error: {str(e)}")
            # Make sure to return the OCR instance to the pool even on error
            if ocr:
                try:
                    self.return_ocr(ocr)
                except:
                    pass  # Ignore errors when returning to the pool
            return []

    def process_image_async(self, image_np, cls=False):
        """
        Process an image with PaddleOCR asynchronously using the thread pool.

        Args:
            image_np: Numpy array of the image
            cls: Whether to use text direction classification

        Returns:
            Future object that will contain the OCR results
        """
        return self.executor.submit(self.process_image, image_np, cls)

    def get_utilization(self):
        """
        Get statistics about pool utilization.
        
        Returns:
            dict: Pool utilization statistics
        """
        with self.access_lock:
            metrics = dict(self.usage_metrics)
            metrics["available"] = self.ocr_queue.qsize()
            
            # Convert float calculations to integers to avoid type errors
            utilization_pct = (
                100 * (self.pool_size - metrics["available"]) / self.pool_size 
                if self.pool_size > 0 else 0
            )
            max_utilization_pct = (
                100 * metrics["max_concurrent"] / self.pool_size 
                if self.pool_size > 0 else 0
            )
            
            # Set calculated metrics
            metrics["utilization_pct"] = int(utilization_pct)
            metrics["max_utilization_pct"] = int(max_utilization_pct)
            
            # Add batch size info as separate keys instead of nested dict
            metrics["max_batch_size"] = self.ocr_params.get("max_batch_size", 1024)
            metrics["rec_batch_num"] = self.ocr_params.get("rec_batch_num", 1024)
            
            return metrics

    def update_batch_sizes(self, max_batch_size=None, rec_batch_num=None):
        """
        Update batch size parameters for OCR processing.
        This doesn't affect currently running instances, only future ones.
        
        Args:
            max_batch_size: New max batch size value
            rec_batch_num: New recognition batch number value
            
        Returns:
            dict: Updated batch size parameters
        """
        with self.access_lock:
            if max_batch_size is not None:
                self.ocr_params["max_batch_size"] = max_batch_size
            
            if rec_batch_num is not None:
                self.ocr_params["rec_batch_num"] = rec_batch_num
            
            return {
                "max_batch_size": self.ocr_params["max_batch_size"],
                "rec_batch_num": self.ocr_params["rec_batch_num"]
            }

    def reset_metrics(self):
        """
        Reset the usage metrics while preserving the current state.
        This is useful for starting a new monitoring session.
        
        Returns:
            dict: Previous metrics before reset
        """
        with self.access_lock:
            current_metrics = dict(self.usage_metrics)
            current_in_use = self.usage_metrics["current_in_use"]
            
            # Reset metrics but keep current_in_use accurate
            self.usage_metrics = {
                "total_requests": 0,
                "max_concurrent": current_in_use,
                "current_in_use": current_in_use
            }
            
            return current_metrics

    def __del__(self):
        """Clean up resources"""
        try:
            # Shutdown the executor
            self.executor.shutdown(wait=False)

            # Empty the queue to release resources
            while not self.ocr_queue.empty():
                try:
                    self.ocr_queue.get_nowait()
                except:
                    break
        except:
            pass  # Ignore errors during cleanup
