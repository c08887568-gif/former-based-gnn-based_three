import logging
import os
import sys
import torch
from thop import profile
from datetime import datetime
import matplotlib.pyplot as plt
import platform
import pkg_resources
import traceback
from utils.utils import WarmupCosineLR
import torch.optim as optim
import atexit
import numpy as np
import pandas as pd
class Logger:
    def __init__(self, model_name, dataset_kind,log_dir=None):
        if log_dir is None:
            current_file_dir = os.path.dirname(__file__)
            parent_dir = os.path.dirname(current_file_dir)
            log_dir = os.path.join(parent_dir, 'logs')
        self.model_name = model_name
        self.log_dir = log_dir
        self.loss_curves_dir = './outputs/'+model_name+'/'+dataset_kind+'/loss_curves'
        self.accuracy_curves_dir = './outputs/'+model_name+'/'+dataset_kind+'/accuracy_curves'
        self.trajectories_dir = './outputs/'+model_name+'/'+dataset_kind+'/trajectories'
        self.RemoteSensingImages_dir = './outputs/'+model_name+'/'+dataset_kind+'/RemoteSensingImages'
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(self.loss_curves_dir, exist_ok=True)
        os.makedirs(self.accuracy_curves_dir, exist_ok=True)
        os.makedirs(self.trajectories_dir, exist_ok=True)
        os.makedirs(self.RemoteSensingImages_dir, exist_ok=True)
        # matplotlib_logger = logging.getLogger('matplotlib')
        # matplotlib_logger.setLevel(logging.WARNING)
        # Record start time
        self.start_time = datetime.now()
        self.filename = os.path.join(log_dir,f'{self.start_time.strftime("%Y%m%d_%H%M%S")}.log')
        self.logger_header = self.setup_logger(
            f'header_{__name__}',
            logging.INFO,
            logging.Formatter(('-' * 80)+'\n'+'%(asctime)s - frs - %(levelname)s - %(message)s'+ '\n' + ('-' * 80))
        )
        self.logger_environment = self.setup_logger(
            f'environment_{__name__}',
            logging.INFO,
            logging.Formatter('')
        )
        self.logger_dataset = self.setup_logger(
            f'dataset_{__name__}',
            logging.INFO,
            logging.Formatter('')
        )
        self.logger_model = self.setup_logger(
            f'model_{__name__}',
            logging.INFO,
            logging.Formatter('')
        )
        self.logger_training_config = self.setup_logger(
            f'training_config_{__name__}',
            logging.INFO,
            logging.Formatter('')
        )
        self.logger_runner = self.setup_logger(
            f'runner_{__name__}',
            logging.INFO,
        )
        self.logger_outputs = self.setup_logger(
            f'outputs_{__name__}',
            logging.INFO,
        )
        self.logger_other = self.setup_logger(
            f'other_{__name__}',
            logging.INFO
        )
        self.logger_header.info("This is your training log file!")
        self.loggers = [self.logger_header,self.logger_environment,self.logger_dataset,self.logger_model
                       ,self.logger_training_config,self.logger_runner,self.logger_outputs,self.logger_other]
        sys.excepthook = self.handle_exception
    def setup_logger(self,name, level=logging.INFO, formatter=None):
        """Function to create and configure a new logger."""
        logger = logging.getLogger(name)
        logger.setLevel(level)

        # 创建一个 Formatter
        if formatter is None:
            formatter = logging.Formatter('%(asctime)s - frs - %(levelname)s - %(message)s')

        # 创建一个 FileHandler
        file_handler = logging.FileHandler(self.filename)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        # 创建一个 StreamHandler
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

        return logger
    def log_environment_info(self):
        """Log environment information."""
        self.logger_header.info("Environment Information:")
        self.logger_environment.info(f"Python Version: {platform.python_version()}")
        self.logger_environment.info("Installed Packages:")
        installed_packages = pkg_resources.working_set
        installed_packages_list = sorted(["%s==%s" % (i.key, i.version)
                                           for i in installed_packages])
        for package in installed_packages_list:
            self.logger_environment.info(package)
    def log_dataset_info(self, train_dataset,valid_dataset,test_dataset):
        """Log dataset information."""
        self.logger_header.info(f"Dataset Information:")
        if train_dataset is not None:
            self.logger_dataset.info(f"Train Dataset: {train_dataset}")
        if valid_dataset is not None:
            self.logger_dataset.info(f"Valid Dataset: {valid_dataset}")
        if test_dataset is not None:
            self.logger_dataset.info(f"Test Dataset: {test_dataset}")
    def log_model_info(self, model):
        """Log model information."""
        self.logger_header.info(f"Model Information:")
        self.logger_model.info(f"Model Name: {self.model_name}")
        self.logger_model.info("Model Structure:")
        """Print the structure of the model."""
        for name, module in model.named_children():
            self.logger_model.info(f"{name}: {module}")
            if hasattr(module, 'extra_repr'):
                extra_repr = module.extra_repr().strip()
                if extra_repr:
                    self.logger_model.info(f"\t{extra_repr}")
        
        macs, params = model.count_flops_and_params()
        self.logger_model.info(f"FLOPs: {macs / 1e9:.2f} GFLOPs")
        self.logger_model.info(f"Parameters: {params / 1e6:.2f} M")
    def log_training_config_info(self,optimizer,scheduler,hyperparameter,train_loader,valid_loader,test_loader):
        """Log training config information."""
        self.logger_header.info(f"Config Information:")
        self.logger_training_config.info(f"total_epochs: {hyperparameter['total_epochs']}")
        self.logger_training_config.info(f"device: {hyperparameter['device']}")
        self.logger_training_config.info(f"Train Loader: {train_loader}")
        self.logger_training_config.info(f"Valid Loader: {valid_loader}")
        self.logger_training_config.info(f"Test Loader: {test_loader}")
        self.logger_training_config.info(f"Optimizer: {optimizer}")
        self.logger_training_config.info(f"Scheduler: {scheduler}")
        
    def log_training_info(self, epoch, learning_rate, train_loss, train_acc, valid_loss, valid_acc, best_valid_acc, train_time, train_fps, valid_time, valid_fps):
        """Log training information."""
        # Log the epoch and learning rate
        self.logger_header.info(f"Epoch {epoch}: , learning_rate: {learning_rate}")
        
        # Prepare strings for train and validation logging
        train_acc_str = f"Train Acc: {train_acc:.4f}, " if train_acc is not None else ""
        valid_acc_str = f"Valid Acc: {valid_acc:.4f}, " if valid_acc is not None else ""
        best_valid_acc_str = f"Best Valid Acc: {best_valid_acc:.4f}, " if best_valid_acc is not None else ""
        
        train_loss_str = f"Train Loss: {train_loss:.4f}, " if train_loss is not None else ""
        valid_loss_str = f"Valid Acc: {valid_loss:.4f}, " if valid_loss is not None else ""
        # Log train information
        self.logger_runner.info(f"{train_loss_str}{train_acc_str}Training Time: {train_time*1000:.4f} ms, Train FPS: {train_fps:.0f}")

        # Log validation information
        self.logger_runner.info(f"{valid_loss_str}{valid_acc_str}{best_valid_acc_str}Validation Time: {valid_time*1000:.4f} ms, Valid FPS: {valid_fps:.0f}")
    
    def log_test_info(self, test_time, test_fps, class_result):
        """Log test information."""
        self.logger_runner.info("Test Classification Metrics:")
        self.logger_runner.info('\n'+class_result)
        self.logger_runner.info(f"Testing Time: {test_time*1000:.4f} ms, Test FPS: {test_fps:.0f}")
    
    def plot_metrics(self, train_losses, valid_losses, train_accuracies=None, valid_accuracies=None,is_save=False):
        """Plot training and validation metrics."""
        if train_losses is not None:
            plt.figure(figsize=(10, 5))
            plt.plot(range(1, len(train_losses)+1), train_losses, label='Training Loss')
            plt.plot(range(1, len(valid_losses)+1), valid_losses, label='Validation Loss')
            plt.title('Training and Validation Loss Over Epochs')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True)
            if is_save is True:
                loss_curve_path = os.path.join(self.loss_curves_dir,f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_loss_plot.png")
                plt.savefig(loss_curve_path)
                self.logger_outputs.info(f"loss_curve: {loss_curve_path}")
            plt.show()
        if train_accuracies is not None:
            plt.figure(figsize=(10, 5))
            plt.plot(range(1, len(train_accuracies)+1), train_accuracies, label='Training Accuracy')
            plt.plot(range(1, len(valid_accuracies)+1), valid_accuracies, label='Validation Accuracy')
            plt.title('Training and Validation Accuracy Over Epochs')
            plt.xlabel('Epochs')
            plt.ylabel('Accuracy')
            plt.legend()
            plt.grid(True)
            if is_save is True:
                accuracy_curve_path = os.path.join(self.accuracy_curves_dir, f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_accuracy_plot.png")
                plt.savefig(accuracy_curve_path)
                self.logger_outputs.info(f"accuracy_curve: {accuracy_curve_path}")
            plt.show()
    def log_predict_info(self,coordinates,predicts,prelabels,labels,trace_id,class_result):
        filename = os.path.basename(trace_id)
        if prelabels.ndim == 1:
            prelabels = prelabels[:, np.newaxis]

        if labels.ndim == 1:
            labels = labels[:, np.newaxis]
            
        if predicts.ndim == 1:
            predicts = predicts[:, np.newaxis]
        # Combine the data into a single numpy array
        combined_data = np.hstack((coordinates,predicts,prelabels, labels))

        # Create a pandas DataFrame
        df = pd.DataFrame(combined_data, columns=['Longitude', 'Latitude', 'Predict','Predicted_Label', 'True_Label'])

        # Save DataFrame to an Excel file
        save_path = os.path.join(self.trajectories_dir,filename)
        df.to_excel(save_path, index=False)
        self.logger_outputs.info(f"predict trajectory: {save_path}"+'\n'+class_result)
    def handle_exception(self,exc_type, exc_value, exc_traceback):
        """Handle exceptions and log them."""
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return

        self.logger_other.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))
        self.clean_up_logger()
        sys.exit(1)
    
    def log_end_of_training(self,avg_train_time,avg_valid_time,avg_test_time,final_class_result_train,final_class_result_valid,
                           final_class_result_test_train,final_class_result_test_valid,final_class_result_pretrain=None):
        """Log the end of training and calculate total training time."""
        end_time = datetime.now()
        total_training_time = end_time - self.start_time
        self.logger_runner.info(f"Avg training time per epoch: {avg_train_time*1000:.4f} ms, Avg validation time per epoch:{avg_valid_time*1000:.4f} ms, Avg testing time per epoch: {avg_test_time*1000:.4f} ms")
        self.logger_other.info(f"Training ended at: {end_time}")
        self.logger_other.info(f"Total training time: {total_training_time}")
        if final_class_result_train is not None:
            self.logger_other.info(f"Train result: \n{final_class_result_train}")
        if final_class_result_valid is not None:
            self.logger_other.info(f"Valid result: \n{final_class_result_valid}")
        if final_class_result_test_train is not None:
            self.logger_other.info(f"Test result (based train): \n{final_class_result_test_train}")
        if final_class_result_test_valid is not None:
            self.logger_other.info(f"Test result (based valid): \n{final_class_result_test_valid}")
        if final_class_result_pretrain is not None:
            self.logger_other.info(f"Pretrain result: \n{final_class_result_pretrain}")
    def log_start_of_outputs(self):
        self.logger_header.info(f"Outputs Information: ")
    def clean_up_logger(self):
        """Close the logger handlers and remove loggers from the manager."""
        try:
            for logger in self.loggers:
                for handler in logger.handlers:
                    handler.close()
                    logger.removeHandler(handler)
                # Remove the logger from the logging module's dict of loggers
                logging.Logger.manager.loggerDict.pop(logger.name, None)
            self.loggers = None
        except:
            pass
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc_value, exc_traceback):
        """Ensure the log files are flushed and closed before cleaning up."""
        # Flush the log files to ensure they are written to disk
        try:
            for logger in self.loggers:
                for handler in logger.handlers:
                    handler.flush()
            # Close the log files
            self.clean_up_logger()
        except:
            pass
    def __del__(self):
        self.clean_up_logger()