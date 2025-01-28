import tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image, ImageTk
import cv2
import os
import numpy as np
from numpy import linalg as la
import statistics as s
import time
import pandas as pd

import ttkbootstrap as ttk
from ttkbootstrap.constants import *

from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from typing import Tuple, List, Optional
from abc import ABC, abstractmethod
from enum import Enum
from typing import List, Optional, Union, Dict
import numpy as np
import numpy.linalg as la
import pandas as pd

class DataManager:
    def __init__(self, data_path: str = "../att_faces"):
        self.data_path = os.path.abspath(data_path)
        self.IMAGE_HEIGHT = 112
        self.IMAGE_WIDTH = 92
        self.VECTOR_SIZE = self.IMAGE_HEIGHT * self.IMAGE_WIDTH

    @staticmethod
    def _extract_subject_number(filename: str) -> int:
        if filename.startswith('s'):
            return int(filename[1:])
        return int(filename.split('.')[0])

    def _process_directory(self, root: str, dirs: List[str], files: List[str], 
                         validation_size: int) -> Tuple[str, List[str], List[str], List[str]]:
        sorted_dirs = sorted([d for d in dirs if d.startswith('s')], 
                           key=self._extract_subject_number)
        
        pgm_files = sorted([f for f in files if f.endswith('.pgm')], 
                          key=self._extract_subject_number)
        
        split_idx = len(pgm_files) - validation_size
        training_files = pgm_files[:split_idx]
        validation_files = pgm_files[split_idx:split_idx + validation_size]
        
        return root, sorted_dirs, training_files, validation_files

    def _load_image(self, image_path: str) -> Optional[np.ndarray]:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            return None
        return image.reshape(self.VECTOR_SIZE)
    
    def get_person_image_path(self, person: int) -> str:
        """Get the path to a person's training image."""
        return os.path.join(self.data_path, f's{person}', '1.pgm')
        

    def prepare_datasets(self, validation_percent: int = 20) -> Tuple[np.ndarray, List[List[str]]]:
        images_per_subject = 10
        validation_per_subject = validation_percent // 10
        training_images = images_per_subject - validation_per_subject
        num_subjects = 40
        
        training_matrix = np.zeros([self.VECTOR_SIZE, training_images * num_subjects])
        validation_files = []
        
        for root, dirs, files in os.walk(self.data_path):
            root, dirs, train_files, valid_files = self._process_directory(
                root, dirs, files, validation_per_subject)
            
            if not train_files:
                continue
                
            if valid_files:
                validation_files.append(valid_files)
            
            for idx, file in enumerate(train_files):
                subject_num = self._extract_subject_number(
                    os.path.basename(os.path.dirname(os.path.join(root, file))).replace('s', ''))
                
                image_vector = self._load_image(os.path.join(root, file))
                if image_vector is not None:
                    col_idx = training_images * (subject_num - 1) + idx
                    training_matrix[:, col_idx] = image_vector
        
        return training_matrix, validation_files

class DistanceMetric(Enum):
    L1 = "l1"
    L2 = "l2"
    INFINITY = "inf"
    COSINE = "cosine"

class FaceRecognitionAlgorithm(ABC):
    def __init__(self, training_matrix: np.ndarray, validation_percent: int):
        self.training_matrix = training_matrix
        self.validation_percent = validation_percent
        self.images_per_person = 10 - validation_percent // 10
        self.results: List[Dict] = []
        
    @abstractmethod
    def recognize(self, test_photo: np.ndarray) -> int:
        pass
    
    @abstractmethod
    def evaluate_and_save(self, validation_data: List[str], data_manager) -> None:
        pass
    
    def calculate_distance(self, 
                         vector1: np.ndarray, 
                         vector2: np.ndarray, 
                         metric: Union[DistanceMetric, str]) -> float:
        if isinstance(metric, str):
            metric = DistanceMetric(metric)
            
        diff = vector1 - vector2
        
        match metric:
            case DistanceMetric.L1:
                return la.norm(diff, 1)
            case DistanceMetric.L2:
                return la.norm(diff, 2)
            case DistanceMetric.INFINITY:
                return la.norm(diff, np.inf)
            case DistanceMetric.COSINE:
                return 1 - np.dot(vector1, vector2) / (
                    la.norm(vector1, 2) * la.norm(vector2, 2)
                )
            case _:
                raise ValueError(f"Unknown norm: {metric}")
    
    def get_person_id(self, index: int) -> int:
        return (index // self.images_per_person) + 1
    
    def save_results(self, filename: str) -> str:
        if not self.results:
            raise ValueError("There are no results to save")
            
        df = pd.DataFrame(self.results)
        df.to_csv(filename, index=False)
        return filename
    
    def add_result(self, result: Dict) -> None:
        self.results.append(result)

class NearestNeighborAlgorithm(FaceRecognitionAlgorithm):
    def __init__(self, training_matrix: np.ndarray, validation_percent: int):
        super().__init__(training_matrix, validation_percent)
        
    def recognize(self, test_image: np.ndarray, norm: Union[DistanceMetric, str]) -> List[int]:
        distances_to_training = np.zeros(len(self.training_matrix[0]))
        
        for training_index in range(self.training_matrix.shape[1]):
            distances_to_training[training_index] = self.calculate_distance(
                self.training_matrix[:,training_index], 
                test_image, 
                norm
            )
            
        closest_match_index = np.argmin(distances_to_training)
        matching_indices = np.where(distances_to_training == distances_to_training[closest_match_index])
        predicted_persons = [self.get_person_id(idx) for idx in matching_indices[0]]
        return predicted_persons  
    
    def evaluate_and_save(self, validation_data: List[str], data_manager) -> str:
        validation_percentages = [10, 20, 40]
        norms = [
            DistanceMetric.L1,          
            DistanceMetric.L2,        
            DistanceMetric.INFINITY,   
            DistanceMetric.COSINE      
        ]
        metric_names = ["manhattan", "euclidean", "infinity", "cosine"]
        
        for validation_percent in validation_percentages:
            training_matrix, current_validation_set = data_manager.prepare_datasets(
                validation_percent=validation_percent
            )
           
            self.training_matrix = training_matrix
            self.validation_percent = validation_percent
            self.images_per_person = 10 - validation_percent//10
            
            for norm, metric_name in zip(norms, metric_names):
                total_time = 0
                correct_predictions = 0
                total_predictions = 0
                
                for person_index, validation_files in enumerate(current_validation_set, 1):
                    start_time = time.time()
                    for image_file in validation_files:
                        test_path = os.path.join(data_manager.data_path, f's{person_index}', image_file)
                        test_image = data_manager._load_image(test_path)
                        
                        predicted_persons = self.recognize(test_image, norm=norm)
                        if person_index in predicted_persons:
                            correct_predictions += 1
                        total_predictions += 1
                        
                    end_time = time.time()
                    total_time += end_time - start_time
                
                self.add_result({
                    'validation_percent': validation_percent,
                    'norm': metric_name,
                    'recognition_rate': (correct_predictions/total_predictions * 100) if total_predictions > 0 else 0,
                    'average_recognition_time': total_time/total_predictions if total_predictions > 0 else 0
                })
        
        return self.save_results(os.path.join(os.getcwd(), 'nearest_neighbor_results.csv'))
    
class KNearestNeighborAlgorithm(FaceRecognitionAlgorithm):
    def __init__(self, training_matrix: np.ndarray, validation_percent: int = 20):
        super().__init__(training_matrix, validation_percent)
        
    def recognize(self, test_image: np.ndarray, norm: Union[DistanceMetric, str] = DistanceMetric.L2, num_neighbors: int = 3) -> int:
        distances_to_training = np.zeros(len(self.training_matrix[0]))
        
        for training_index in range(self.training_matrix.shape[1]):
            distances_to_training[training_index] = self.calculate_distance(
                self.training_matrix[:,training_index], 
                test_image, 
                norm
            )
        
        sorted_distance_indices = np.argsort(distances_to_training)
        nearest_neighbor_indices = sorted_distance_indices[:num_neighbors]
        neighbor_person_ids = [self.get_person_id(idx) for idx in nearest_neighbor_indices]
        
        return s.mode(neighbor_person_ids)
    
    def evaluate_and_save(self, validation_data: List[str], data_manager) -> str:
        validation_percentages = [10, 20, 40]
        norms = [
            DistanceMetric.L1,          
            DistanceMetric.L2,          
            DistanceMetric.INFINITY,    
            DistanceMetric.COSINE       
        ]
        metric_names = ["manhattan", "euclidean", "infinity", "cosine"]
        neighbor_counts = [3, 5, 9]
        
        for num_neighbors in neighbor_counts:
            for validation_percent in validation_percentages:
                training_matrix, current_validation_set = data_manager.prepare_datasets(
                    validation_percent=validation_percent
                )
                
                self.training_matrix = training_matrix
                self.validation_percent = validation_percent
                self.images_per_person = 10 - validation_percent//10
                
                for norm, metric_name in zip(norms, metric_names):
                    total_time = 0
                    correct_predictions = 0
                    total_predictions = 0
                    
                    for person_index, validation_files in enumerate(current_validation_set, 1):
                        start_time = time.time()
                        for image_file in validation_files:
                            test_path = os.path.join(data_manager.data_path, f's{person_index}', image_file)
                            test_image = data_manager._load_image(test_path)
                            
                            predicted_person = self.recognize(
                                test_image, 
                                norm=norm, 
                                num_neighbors=num_neighbors
                            )
                            if predicted_person == person_index:
                                correct_predictions += 1
                            total_predictions += 1
                            
                        total_time += time.time() - start_time
                    
                    self.add_result({
                        'num_neighbors': num_neighbors,
                        'validation_percent': validation_percent,
                        'norm': metric_name,
                        'recognition_rate': (correct_predictions/total_predictions * 100) if total_predictions > 0 else 0,
                        'average_recognition_time': total_time/total_predictions if total_predictions > 0 else 0
                    })
        
        return self.save_results(os.path.join(os.getcwd(), 'knn_recognition_results.csv'))
    
class EigenfacesAlgorithm(FaceRecognitionAlgorithm):
    def __init__(self, training_matrix: np.ndarray, validation_percent: int = 20, use_class_representatives: bool = False):
        super().__init__(training_matrix, validation_percent)
        self.face_mean_vector = None
        self.centered_faces = None  
        self.eigenface_matrix = None  
        self.face_projections = None
        self.class_representatives = None
        self.use_class_representatives = use_class_representatives
        self.preprocessing_time = None
        
    def preprocess(self, num_components: Optional[int] = None) -> None:
        start_time = time.time()
        
        if self.use_class_representatives:
            self._preprocess_with_representatives(num_components)
        else:
            self._preprocess_standard(num_components)
            
        self.preprocessing_time = time.time() - start_time
        
    def _preprocess_standard(self, num_components: Optional[int] = None) -> None:
        self.face_mean_vector = np.mean(self.training_matrix, axis=1)
        self.centered_faces = self.training_matrix - self.face_mean_vector.reshape(-1, 1)
        
        covariance_matrix = np.dot(self.training_matrix.T, self.training_matrix)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
        
        sorted_indices = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[sorted_indices]
        eigenvectors = eigenvectors[:, sorted_indices]
        
        if num_components is not None:
            eigenvectors = eigenvectors[:, :num_components]
            eigenvalues = eigenvalues[:num_components]
        
        self.eigenface_matrix = np.dot(self.training_matrix, eigenvectors)
        self.eigenface_matrix = self.eigenface_matrix / np.sqrt(eigenvalues)
        self.face_projections = np.dot(self.centered_faces.T, self.eigenface_matrix).T
        
    def _preprocess_with_representatives(self, num_components: Optional[int] = None) -> None:
        images_per_class = self.images_per_person
        total_samples = self.training_matrix.shape[1]
        num_classes = total_samples // images_per_class
        
        self.class_representatives = np.zeros((self.training_matrix.shape[0], num_classes))
        
        for class_index in range(num_classes):
            start_index = class_index * images_per_class
            end_index = start_index + images_per_class
            
            if end_index <= total_samples:
                class_images = self.training_matrix[:, start_index:end_index]
                self.class_representatives[:, class_index] = np.mean(class_images, axis=1)
        
        self.face_mean_vector = np.mean(self.class_representatives, axis=1)
        self.centered_faces = self.class_representatives - self.face_mean_vector.reshape(-1, 1)
        covariance_matrix = np.dot(self.centered_faces.T, self.centered_faces)
        
        eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
        sorted_indices = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[sorted_indices]
        eigenvectors = eigenvectors[:, sorted_indices]
        
        if num_components is not None:
            eigenvectors = eigenvectors[:, :num_components]
            eigenvalues = eigenvalues[:num_components]
        
        self.eigenface_matrix = np.dot(self.centered_faces, eigenvectors)
        
        for i in range(self.eigenface_matrix.shape[1]):
            if eigenvalues[i] > 0:  
                self.eigenface_matrix[:, i] = self.eigenface_matrix[:, i] / np.sqrt(eigenvalues[i])
        
        self.face_projections = np.dot(self.centered_faces.T, self.eigenface_matrix).T
        
    def recognize(self, test_image: np.ndarray, metric: Union[DistanceMetric, str] = DistanceMetric.L2) -> int:
        if any(x is None for x in [self.face_mean_vector, self.centered_faces, self.eigenface_matrix, self.face_projections]):
            raise ValueError("Must run preprocess() before recognition")
            
        normalized_image = test_image - self.face_mean_vector
        projected_image = np.dot(self.eigenface_matrix, np.dot(normalized_image.T, self.eigenface_matrix).T)
        eigenfaces_projection = np.dot(normalized_image.T, self.eigenface_matrix)
        
        distances = np.zeros(len(self.face_projections[0]))
        for i in range(self.face_projections.shape[1]):
            distances[i] = self.calculate_distance(self.face_projections[:,i], eigenfaces_projection, metric)
            
        closest_match_index = np.argmin(distances)
        if self.use_class_representatives:
            return closest_match_index + 1   
        return self.get_person_id(closest_match_index)
    
    def evaluate_and_save(self, validation_data: List[str], data_manager) -> str:
        validation_percentages = [10, 20, 40]
        norms = [
            DistanceMetric.L1, 
            DistanceMetric.L2, 
            DistanceMetric.INFINITY,
            DistanceMetric.COSINE
        ]
        metric_names = {
            DistanceMetric.L1: "manhattan",
            DistanceMetric.L2: "euclidean",
            DistanceMetric.INFINITY: "infinity",
            DistanceMetric.COSINE: "cosine"
        }
        component_counts = [20, 40, 60, 80, 100]
        
        for num_components in component_counts:
            for validation_percent in validation_percentages:
                training_matrix, current_validation_set = data_manager.prepare_datasets(
                    validation_percent=validation_percent
                )
               
                self.training_matrix = training_matrix
                self.validation_percent = validation_percent
                self.images_per_person = 10 - validation_percent//10
                
                for norm in norms:
                    self.preprocess(num_components=num_components)                    
                    total_time = 0
                    correct_predictions = 0
                    total_predictions = 0
                    
                    for person_index, validation_files in enumerate(current_validation_set, 1):
                        start_time = time.time()
                        for image_file in validation_files:
                            test_path = os.path.join(data_manager.data_path, f's{person_index}', image_file)
                            test_image = data_manager._load_image(test_path)
                            
                            predicted_person = self.recognize(test_image, metric=norm)
                            if predicted_person == person_index:
                                correct_predictions += 1
                            total_predictions += 1
                            
                        total_time += time.time() - start_time
                    
                    self.add_result({
                        'num_components': num_components,
                        'validation_percent': validation_percent,
                        'norm': metric_names[norm],
                        'preprocessing_time': self.preprocessing_time,
                        'recognition_rate': (correct_predictions/total_predictions * 100) if total_predictions > 0 else 0,
                        'average_recognition_time': total_time/total_predictions if total_predictions > 0 else 0
                    })
        
        results_filename = 'eigenfaces_with_representatives_results.csv' if self.use_class_representatives else 'eigenfaces_standard_results.csv'
        return self.save_results(os.path.join(os.getcwd(), results_filename))
    
class LanczosAlgorithm(FaceRecognitionAlgorithm):
    def __init__(self, training_matrix: np.ndarray, validation_percent: int = 20):
        super().__init__(training_matrix, validation_percent)
        self.face_mean_vector = None
        self.centered_faces = None   
        self.basis_matrix = None    
        self.face_projections = None
        self.preprocessing_time = None

    def lanczos_with_projections(self, input_matrix: np.ndarray, initial_vector: np.ndarray, num_iterations: int) -> Tuple[np.ndarray, np.ndarray]:
        matrix_size = input_matrix.shape[0]
        beta_values = [0.0]
        orthogonal_vectors = [np.zeros(matrix_size), initial_vector]
        basis_vectors = [initial_vector]
        
        for iteration in range(1, num_iterations + 1):
           
            direction_vector = input_matrix @ (input_matrix.T @ orthogonal_vectors[iteration]) - beta_values[iteration-1] * orthogonal_vectors[iteration-1]
            alpha_current = np.dot(direction_vector, orthogonal_vectors[iteration])
            direction_vector = direction_vector - alpha_current * orthogonal_vectors[iteration]
            
            beta_next = np.linalg.norm(direction_vector)
            if np.isclose(beta_next, 0):
                break
                
            beta_values.append(beta_next)
            next_vector = direction_vector / beta_next
            orthogonal_vectors.append(next_vector)
            basis_vectors.append(next_vector)
        
        basis_matrix = np.column_stack(basis_vectors)
        matrix_projections = basis_matrix.T @ input_matrix
        
        return basis_matrix, matrix_projections

    def lanczos_preprocessing(self, input_matrix: np.ndarray, num_components: int = 20) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
        start_time = time.time()
        
        face_mean_vector = np.mean(input_matrix, axis=1)
        centered_faces = input_matrix - face_mean_vector.reshape(-1, 1)
        
        initial_vector = centered_faces[:, 0].copy()
        initial_vector = initial_vector / np.linalg.norm(initial_vector)
        
        basis_matrix, face_projections = self.lanczos_with_projections(centered_faces, initial_vector, num_components)
        
        preprocessing_time = time.time() - start_time
        return face_mean_vector, centered_faces, basis_matrix, face_projections, preprocessing_time

    def nearest_neighbor_search(self, face_projections: np.ndarray, test_projection: np.ndarray, 
                    validation_percent: int = 20, norm: Union[DistanceMetric, str] = DistanceMetric.L1) -> int:
        distances = np.zeros(face_projections.shape[1])
        for index in range(face_projections.shape[1]):
            distances[index] = self.calculate_distance(face_projections[:,index], test_projection, norm)
            
        return np.argmin(distances)
        
    def preprocess(self, num_components: int = 20) -> None:
        self.face_mean_vector, self.centered_faces, self.basis_matrix, self.face_projections, self.preprocessing_time = self.lanczos_preprocessing(
            self.training_matrix, num_components=num_components
        )

    def recognize(self, test_image: np.ndarray, metric: Union[DistanceMetric, str] = DistanceMetric.L2) -> int:
        if any(x is None for x in [self.face_mean_vector, self.centered_faces, self.basis_matrix, self.face_projections]):
            raise ValueError("Must run preprocess() before recognition")
            
        normalized_test_image = test_image - self.face_mean_vector
        test_projection = self.basis_matrix.T @ normalized_test_image
        
        closest_match_index = self.nearest_neighbor_search(
            self.face_projections, 
            test_projection,
            validation_percent=self.validation_percent,
            norm=metric  
        )
        
        return self.get_person_id(closest_match_index)

    def evaluate_and_save(self, validation_data: List[str], data_manager) -> str:
        validation_percentages = [10, 20, 40]
        norms = [
            DistanceMetric.L1,
            DistanceMetric.L2,
            DistanceMetric.INFINITY,
            DistanceMetric.COSINE
        ]
        metric_names = {
            DistanceMetric.L1: "manhattan",
            DistanceMetric.L2: "euclidean",
            DistanceMetric.INFINITY: "infinity",
            DistanceMetric.COSINE: "cosine"
        }
        component_counts = [20, 40, 60, 80, 100]
        
        for num_components in component_counts:
            for validation_percent in validation_percentages:
                training_matrix, current_validation_set = data_manager.prepare_datasets(
                    validation_percent=validation_percent
                )
               
                self.training_matrix = training_matrix
                self.validation_percent = validation_percent
                self.images_per_person = 10 - validation_percent//10
                
                for norm in norms:
                    self.preprocess(num_components=num_components)
                    
                    total_time = 0
                    correct_predictions = 0
                    total_predictions = 0
                    
                    for person_index, validation_files in enumerate(current_validation_set, 1):
                        start_time = time.time()
                        for image_file in validation_files:
                            test_path = os.path.join(data_manager.data_path, f's{person_index}', image_file)
                            test_image = data_manager._load_image(test_path)
                            
                            predicted_person = self.recognize(test_image, metric=norm)
                            if predicted_person == person_index:
                                correct_predictions += 1
                            total_predictions += 1
                            
                        total_time += time.time() - start_time
                    
                    self.add_result({
                        'num_components': num_components,
                        'validation_percent': validation_percent,
                        'norm': metric_names[norm],
                        'preprocessing_time': self.preprocessing_time,
                        'recognition_rate': (correct_predictions/total_predictions * 100) if total_predictions > 0 else 0,
                        'average_recognition_time': total_time/total_predictions if total_predictions > 0 else 0
                    })
        
        return self.save_results(os.path.join(os.getcwd(), 'lanczos_results.csv'))
    

class AlgorithmFactory:
    @staticmethod
    def create_algorithm(algorithm_name: str, 
                        training_matrix: np.ndarray, 
                        validation_percent: int) -> FaceRecognitionAlgorithm:
        algorithms = {
            "NN": NearestNeighborAlgorithm,
            "kNN": KNearestNeighborAlgorithm,
            "Eigenfaces": lambda m, v: EigenfacesAlgorithm(m, v, use_class_representatives=False),
            "Eigenfaces_Representatives": lambda m, v: EigenfacesAlgorithm(m, v, use_class_representatives=True),
            "Lanczos": LanczosAlgorithm
        }
        
        if algorithm_name not in algorithms:
            raise ValueError(f"Unknown algorithm: {algorithm_name}")
            
        return algorithms[algorithm_name](training_matrix, validation_percent)
    
class FaceRecognitionUI:
    FIXED_IMAGE_WIDTH = 300
    FIXED_IMAGE_HEIGHT = 350
    
    def __init__(self, root):
        self.root = root
        self.root.style = ttk.Style(theme='darkly')
        self.initialize_variables()
        self.create_main_layout()   
        self.create_all_sections()
        self.load_training_data()


    def initialize_variables(self):
        self.current_image = None
        self.training_matrix = None
        self.validation_data = None
        self.preprocessed_data = None
        self.k_entries = {}
        self.results = []
        self.data_manager = DataManager(os.path.join(os.getcwd(), "att_faces"))

        
        self.splits_map = {
            "90% training 10% testing": 10,
            "80% training 20% testing": 20,
            "60% training 40% testing": 40
        }
        
        self.metric_map = {
            "Manhattan": DistanceMetric.L1,
            "Euclidean": DistanceMetric.L2, 
            "Infinity": DistanceMetric.INFINITY,  
            "Cosine": DistanceMetric.COSINE    
        }
        
        self.metric_names = {
            DistanceMetric.L1: "manhattan",
            DistanceMetric.L2: "euclidean",    
            DistanceMetric.INFINITY: "infinity", 
            DistanceMetric.COSINE: "cosine"     
        }
        
        self.db_var = tk.StringVar(value="ORL")
        self.split_var = tk.StringVar(value="80% training 20% testing")
        self.alg_var = tk.StringVar(value="NN")
        self.norm_var = tk.StringVar(value="Euclidean")
        self.save_alg_var = tk.StringVar(value="NN")
        self.stats_var = tk.StringVar(value="NN")

    def create_main_layout(self):
        """Create main window layout"""
        self.main_container = ttk.Frame(self.root)
        self.main_container.pack(fill=tk.BOTH, expand=True)
        
       
        self.control_frame = ttk.Frame(self.main_container, width=300)
        self.control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        self.control_frame.pack_propagate(False)  
        
        self.image_frame = ttk.Frame(self.main_container)
        self.image_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.create_image_frame()
    
    def create_image_frame(self):
        """Create and setup the image display area"""
        self.image_display_frame = ttk.Frame(self.image_frame)
        self.image_display_frame.pack(expand=True, fill=tk.BOTH)
        
        self.create_original_image_frame()
        self.create_matched_image_frame()

    def create_original_image_frame(self):
        self.original_image_frame = ttk.Frame(self.image_display_frame)
        self.original_image_frame.pack(side=tk.LEFT, expand=True, fill=tk.BOTH, padx=10)
        
        ttk.Label(self.original_image_frame, text="Original Image").pack()
        
        self.image_label = ttk.Label(self.original_image_frame)
        self.image_label.pack(expand=True, fill=tk.BOTH, pady=10)
    
    def create_matched_image_frame(self):
        self.matched_image_frame = ttk.Frame(self.image_display_frame)
        self.matched_image_frame.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH, padx=10)
        
        ttk.Label(self.matched_image_frame, text="Found Image").pack()
        
        self.matched_label = ttk.Label(self.matched_image_frame)
        self.matched_label.pack(expand=True, fill=tk.BOTH, pady=10)

    def create_all_sections(self):
        """Create all control sections"""
        self.create_database_section()
        self.create_configuration_section()
        self.create_algorithm_section()
        self.create_norm_section()
        self.create_file_selection_section()
        self.create_action_buttons_section()
        self.create_save_section()

    def create_database_section(self):
        frame = ttk.LabelFrame(self.control_frame, text="DATABASE")
        frame.pack(padx=10, pady=5, fill="x")
        
        for text in ["ORL", "Essex", "CTOWF"]:
            ttk.Radiobutton(frame, text=text, variable=self.db_var, value=text).pack(anchor=tk.W)

    def create_configuration_section(self):
        frame = ttk.LabelFrame(self.control_frame, text="DATABASE CONFIGURATION")
        frame.pack(padx=10, pady=5, fill="x")
        
        for text in self.splits_map.keys():
            ttk.Radiobutton(frame, text=text, variable=self.split_var,
                          value=text, command=self.load_training_data).pack(anchor=tk.W)

    def create_algorithm_section(self):
        frame = ttk.LabelFrame(self.control_frame, text="ALGORITHM")
        frame.pack(padx=10, pady=5, fill="x")
        
        algorithms = ["NN", "kNN", "Eigenfaces", "Eigenfaces cu RC", "Lanczos", "Tensor"]
        
        for alg in algorithms:
            self.create_algorithm_row(frame, alg)

    def create_algorithm_row(self, parent_frame, algorithm):
        alg_frame = ttk.Frame(parent_frame)
        alg_frame.pack(fill="x")
        
        ttk.Radiobutton(alg_frame, text=algorithm, variable=self.alg_var,
                       value=algorithm).pack(side=tk.LEFT)
        
        if algorithm not in ["NN", "Tensor"]:
            ttk.Label(alg_frame, text="k=").pack(side=tk.LEFT)
            k_entry = ttk.Entry(alg_frame, width=5)
            k_entry.pack(side=tk.LEFT)
            k_entry.insert(0, "3")
            self.k_entries[algorithm] = k_entry

    def create_norm_section(self):
        frame = ttk.LabelFrame(self.control_frame, text="NORM")
        frame.pack(padx=10, pady=5, fill="x")
        
        norms = ["Manhattan", "Euclidean", "Infinity", "Cosine"]   
        for norm in norms:
            ttk.Radiobutton(frame, text=norm, variable=self.norm_var,
                          value=norm).pack(anchor=tk.W)


    def create_file_selection_section(self):
        """Create file selection section"""
        btn_frame = ttk.Frame(self.control_frame)
        btn_frame.pack(padx=10, pady=5, fill="x")
        
        ttk.Button(btn_frame, text="Choose File",
                  command=self.choose_file).pack(side=tk.LEFT, padx=5)
        self.file_label = ttk.Label(btn_frame, text="No file selected")
        self.file_label.pack(side=tk.LEFT, padx=5)

    def create_action_buttons_section(self):
        """Create action buttons section"""
        btn_frame = ttk.Frame(self.control_frame)
        btn_frame.pack(padx=10, pady=5, fill="x")
        
        ttk.Button(btn_frame, text="Preprocess",
                  command=self.preprocess).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Search",
                  command=self.search_face).pack(side=tk.LEFT, padx=5)

    def create_save_section(self):
        """Create save results and statistics section"""
       
        control_section = ttk.Frame(self.control_frame)
        control_section.pack(padx=10, pady=5, fill="x")
        
       
        save_frame = ttk.LabelFrame(control_section, text="Save Results")
        save_frame.pack(fill="x", pady=5)
        
        algorithms = ["NN", "kNN", "Eigenfaces", "Eigenfaces cu RC", "Lanczos", "Tensor"]
        save_dropdown = ttk.Combobox(save_frame,
                                   textvariable=self.save_alg_var,
                                   values=algorithms,
                                   state="readonly",
                                   width=15)
        save_dropdown.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(save_frame, text="Save Results",
                  command=self.save_results).pack(side=tk.LEFT, padx=5)
        
        
        stats_frame = ttk.LabelFrame(control_section, text="Statistics")
        stats_frame.pack(fill="x", pady=5)
        
        stats_dropdown = ttk.Combobox(stats_frame,
                                    textvariable=self.stats_var,
                                    values=algorithms,
                                    state="readonly",
                                    width=15)
        stats_dropdown.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(stats_frame, text="Show Statistics",
                  command=self.show_statistics).pack(side=tk.LEFT, padx=5)
        
        self.result_label = ttk.Label(self.control_frame, text="")
        self.result_label.pack(padx=10, pady=5)

    def load_training_data(self):
        """Load training data based on current configuration"""
        validation_percent = self.splits_map[self.split_var.get()]
        self.training_matrix, self.validation_data = self.data_manager.prepare_datasets(
            validation_percent=validation_percent
        )
        self.preprocessed_data = None 

    def choose_file(self):
        """Handle file selection"""
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.gif *.bmp *.tiff *.pgm")]
        )
        if file_path:
            self.file_label.config(text=file_path.split('/')[-1])
            self.display_image(file_path)
            self.current_image = self.prepare_image(file_path)

    def prepare_image(self, file_path):
        """Prepare image for processing"""
        img = cv2.imread(file_path, 0)
        if img is not None:
            img = cv2.resize(img, (92, 112))
            return img.reshape(10304,)
        return None

    def search_face(self):
        if self.current_image is None:
            self.result_label.config(text="Please select an image first")
            return
        
        algorithm_name = self.alg_var.get()
        metric_name = self.norm_var.get()
        validation_percent = self.splits_map[self.split_var.get()]
        metric = self.metric_map[metric_name]
        
        try:
            start_time = time.time()
            
            if algorithm_name in ["Eigenfaces", "Eigenfaces cu RC", "Lanczos"]:
                if self.preprocessed_data is None:
                    self.result_label.config(text="Please preprocess data first")
                    return
                algorithm = self.preprocessed_data
            else:
                algorithm_map = {
                    "Eigenfaces": "Eigenfaces",
                    "Eigenfaces cu RC": "Eigenfaces_Representatives",
                    "Lanczos": "Lanczos"
                }
                factory_algorithm_name = algorithm_map.get(algorithm_name, algorithm_name)
                algorithm = AlgorithmFactory.create_algorithm(
                    factory_algorithm_name,
                    self.training_matrix,
                    validation_percent
                )
            
            if algorithm_name == "kNN":
                num_neighbors = self.get_k_value(algorithm_name)
                person = algorithm.recognize(self.current_image, norm=metric, num_neighbors=num_neighbors)
            else:
                if algorithm_name in ["Eigenfaces", "Eigenfaces cu RC", "Lanczos"]:
                    person = algorithm.recognize(self.current_image, metric=metric)
                else:
                    person = algorithm.recognize(self.current_image, norm=metric)[0]
            
            execution_time = time.time() - start_time
            
            self.store_result(algorithm_name, validation_percent, self.metric_names[metric], 
                            person, execution_time)
            self.display_result(person, execution_time)
            
        except Exception as e:
            print(f"Error during search: {str(e)}")
            self.result_label.config(text=f"Error during search: {str(e)}")


    def store_result(self, algorithm, validation_percent, norm_type, person, execution_time):
        """Store recognition results"""
        result_data = {
            'algorithm': algorithm,
            'validation_percentage': validation_percent,
            'norm': norm_type,
            'predicted_person': person,
            'execution_time': execution_time,
            'k_neighbors': self.get_k_value(algorithm) if algorithm == "kNN" else None
        }
        self.results.append(result_data)

    def display_result(self, person, execution_time):
        """Display recognition results"""
        self.result_label.config(text=f"Predicted person: {person} (Time: {execution_time:.3f}s)")
        found_person_path = os.path.join(self.data_manager.data_path, f's{person}', '1.pgm')

        self.display_matched_image(found_person_path)

    def get_k_value(self, algorithm):
        """Get k value for specified algorithm"""
        if algorithm in self.k_entries:
            try:
                return int(self.k_entries[algorithm].get())
            except ValueError:
                return 3
        return 3

    def preprocess(self):
        algorithm = self.alg_var.get()
        if algorithm in ["Eigenfaces", "Eigenfaces cu RC", "Lanczos"]:
            try:
                k = self.get_k_value(algorithm)
                validation_percent = self.splits_map[self.split_var.get()]
                
                
                algorithm_map = {
                    "Eigenfaces": "Eigenfaces", 
                    "Eigenfaces cu RC": "Eigenfaces_Representatives",
                    "Lanczos": "Lanczos"
                }
                
                
                factory_algorithm_name = algorithm_map.get(algorithm)
                if factory_algorithm_name is None:
                    raise ValueError(f"Unknown algorithm: {algorithm}")
                    
                print(f"Creating algorithm: {factory_algorithm_name}")  
                
              
                self.preprocessed_data = AlgorithmFactory.create_algorithm(
                    factory_algorithm_name,
                    self.training_matrix,
                    validation_percent
                )
                
                print(f"Preprocessing with k={k}")  
                self.preprocessed_data.preprocess(num_components=k)
                
                self.result_label.config(text="Preprocessing completed")
            except Exception as e:
                print(f"Full preprocessing error: {str(e)}")  
                self.result_label.config(text=f"Preprocessing error: {str(e)}")


    def save_results(self):
        selected_algorithm = self.save_alg_var.get()
        validation_percent = self.splits_map[self.split_var.get()]
        
        try:
            algorithm_map = {
                "Eigenfaces": "Eigenfaces",
                "Eigenfaces cu RC": "Eigenfaces_Representatives",
                "Lanczos": "Lanczos"
            }
            
            factory_algorithm_name = algorithm_map.get(selected_algorithm, selected_algorithm)
            algorithm = AlgorithmFactory.create_algorithm(
                factory_algorithm_name,
                self.training_matrix,
                validation_percent
            )
            
            if selected_algorithm in ["Eigenfaces", "Eigenfaces cu RC", "Lanczos"]:
                k = self.get_k_value(selected_algorithm)
                algorithm.preprocess(num_components=k)
                
            saved_file = algorithm.evaluate_and_save(self.validation_data, self.data_manager)
            self.result_label.config(text=f"Results saved to {saved_file}")
                    
        except Exception as e:
            print(f"Error saving results: {str(e)}")
            self.result_label.config(text=f"Error saving results: {str(e)}")

    def display_image(self, file_path):
        """Display image in the UI"""
        if not file_path:
            return
        
       
        image = Image.open(file_path)
        resized_image = image.resize((self.FIXED_IMAGE_WIDTH, self.FIXED_IMAGE_HEIGHT), Image.Resampling.LANCZOS)
        
        photo = ImageTk.PhotoImage(resized_image)
        
        self.image_label.image = photo 
        self.image_label.configure(image=photo)
    
    def display_matched_image(self, file_path):
        """Display matched image in the UI"""
        if not file_path:
            return
        
        try:
           
            image = Image.open(file_path)
            resized_image = image.resize((self.FIXED_IMAGE_WIDTH, self.FIXED_IMAGE_HEIGHT), Image.Resampling.LANCZOS)
            
           
            photo = ImageTk.PhotoImage(resized_image)
            
           
            self.matched_label.image = photo  
            self.matched_label.configure(image=photo)
        except Exception as e:
            print(f"Error displaying matched image: {str(e)}")
            
            
    def show_statistics(self):
        """Afișează toate statisticile disponibile pentru algoritmul selectat"""
        algorithm = self.stats_var.get()
        
      
        stats_window = tk.Toplevel(self.root)
        stats_window.title(f"Statistics for {algorithm}")
        stats_window.geometry("1200x800")
        
      
        main_frame = ttk.Frame(stats_window)
        main_frame.pack(fill=tk.BOTH, expand=True)
        canvas = tk.Canvas(main_frame)
        scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        algorithm_map = {
           "NN": "nearest_neighbor_results.csv",
           "kNN": "knn_recognition_results.csv",
           "Eigenfaces": "eigenfaces_standard_results.csv", 
           "Eigenfaces cu RC": "eigenfaces_with_representatives_results.csv",
           "Lanczos": "lanczos_results.csv"
        }
        
        try:
            file_name = algorithm_map.get(algorithm)
            if not file_name:
                raise ValueError(f"No file mapping for algorithm {algorithm}")
            
            file_path = os.path.join(os.getcwd(), file_name)
            df = pd.read_csv(file_path)
            
            if algorithm == "NN":
                self._create_nn_plots(df, scrollable_frame)
            elif algorithm == "kNN":
                self._create_knn_plots(df, scrollable_frame)
            elif algorithm in ["Eigenfaces", "Eigenfaces cu RC", "Lanczos"]:
                self._create_projective_algorithm_plots(df, scrollable_frame, algorithm)
            else:
                raise ValueError(f"Unsupported algorithm: {algorithm}")
        except Exception as e:
            error_label = ttk.Label(scrollable_frame, text=f"Error loading statistics: {str(e)}", foreground="red")
            error_label.pack(pady=20)
    
   
    def _create_nn_plots(self, df, parent_frame):
        """Creates visualizations for NN algorithm analyzing each validation percentage"""
        validation_pcts = sorted(df['validation_percent'].unique())
        norms = ['manhattan', 'euclidean', 'infinity', 'cosine']
        
       
        for val_pct in validation_pcts:
            val_frame = ttk.LabelFrame(parent_frame, text=f"Analysis for {val_pct}% Validation Data")
            val_frame.pack(pady=5, padx=5, fill='x')
            
          
            fig = plt.figure(figsize=(10, 4))  
            
            
            ax1 = plt.subplot(121)
            mask = df['validation_percent'] == val_pct
            rates = []
            for norm in norms:
                norm_mask = mask & (df['norm'] == norm)
                rate = df[norm_mask]['recognition_rate'].mean()
                rates.append(rate)
            
            bars1 = ax1.bar(norms, rates, color='skyblue')
            ax1.set_title('Recognition Rate')
            ax1.set_xlabel('Norm')
            ax1.set_ylabel('Rate')
            plt.xticks(rotation=45)
            ax1.grid(True, axis='y')
            
            
            for bar in bars1:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.3f}', ha='center', va='bottom')
            
            ax2 = plt.subplot(122)
            times = []
            for norm in norms:
                norm_mask = mask & (df['norm'] == norm)
                time = df[norm_mask]['average_recognition_time'].mean()
                times.append(time)
            
            bars2 = ax2.bar(norms, times, color='lightcoral')
            ax2.set_title('Average Query Time')
            ax2.set_xlabel('Norm')
            ax2.set_ylabel('Time (s)')
            plt.xticks(rotation=45)
            ax2.grid(True, axis='y')
            
            
            for bar in bars2:
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.4f}', ha='center', va='bottom')
            
            plt.tight_layout()
            canvas = FigureCanvasTkAgg(fig, master=val_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(pady=5, padx=5)
            
            
            stats_text = f"""
            Statistics for {val_pct}% validation data:
            Recognition Rates: """ + " | ".join([f"{norm}: {rate:.3f}%" for norm, rate in zip(norms, rates)]) + """
            Average Query Times: """ + " | ".join([f"{norm}: {time:.4f}s" for norm, time in zip(norms, times)])
            
            stats_label = ttk.Label(val_frame, text=stats_text, justify='left', font=('Courier', 9))
            stats_label.pack(pady=2, padx=5, anchor='w')
        
         
        comparison_frame = ttk.LabelFrame(parent_frame, text="Overall Comparison")
        comparison_frame.pack(pady=5, padx=5, fill='x')
        
        fig = plt.figure(figsize=(10, 4))  
        
        
        ax1 = plt.subplot(121)
        for norm in norms:
            rates = []
            for val_pct in validation_pcts:
                mask = (df['validation_percent'] == val_pct) & (df['norm'] == norm)
                rates.append(df[mask]['recognition_rate'].mean())
            ax1.plot(validation_pcts, rates, marker='o', label=norm)
        
        ax1.set_title('Recognition Rate by Validation Size')
        ax1.set_xlabel('Validation %')
        ax1.set_ylabel('Rate')
        ax1.legend(loc='best', fontsize='small')
        ax1.grid(True)
         
        ax2 = plt.subplot(122)
        for norm in norms:
            times = []
            for val_pct in validation_pcts:
                mask = (df['validation_percent'] == val_pct) & (df['norm'] == norm)
                times.append(df[mask]['average_recognition_time'].mean())
            ax2.plot(validation_pcts, times, marker='o', label=norm)
        
        ax2.set_title('Average Query Time by Validation Size')
        ax2.set_xlabel('Validation %')
        ax2.set_ylabel('Time (s)')
        ax2.legend(loc='best', fontsize='small')
        ax2.grid(True)
        
        plt.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=comparison_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(pady=5, padx=5)
        
        
        overall_stats = f"""Best Recognition Rate: {df['recognition_rate'].max():.3f}% (Norm: {df.loc[df['recognition_rate'].idxmax(), 'norm']}, Val: {df.loc[df['recognition_rate'].idxmax(), 'validation_percent']}%)
        Fastest Configuration: {df['average_recognition_time'].min():.4f}s (Norm: {df.loc[df['average_recognition_time'].idxmin(), 'norm']}, Val: {df.loc[df['average_recognition_time'].idxmin(), 'validation_percent']}%)"""
        
        overall_stats_label = ttk.Label(comparison_frame, text=overall_stats, justify='left', font=('Courier', 9))
        overall_stats_label.pack(pady=2, padx=5, anchor='w')

    def _create_knn_plots(self, df, parent_frame):
        """Creates visualizations for kNN algorithm analyzing each k value and validation percentage"""
        validation_pcts = sorted(df['validation_percent'].unique())
        k_values = sorted(df['num_neighbors'].unique())
        norms = ['manhattan', 'euclidean', 'infinity', 'cosine']
        
        
        for k in k_values:
            k_frame = ttk.LabelFrame(parent_frame, text=f"Analysis for k={k}")
            k_frame.pack(pady=5, padx=5, fill='x')
            
            fig = plt.figure(figsize=(10, 4))
            
            
            ax1 = plt.subplot(121)
            mask = df['num_neighbors'] == k
            rates = []
            for norm in norms:
                norm_mask = mask & (df['norm'] == norm)
                rate = df[norm_mask]['recognition_rate'].mean()
                rates.append(rate)
            
            bars1 = ax1.bar(norms, rates, color='skyblue')
            ax1.set_title('Recognition Rate')
            ax1.set_xlabel('Norm')
            ax1.set_ylabel('Rate')
            plt.xticks(rotation=45)
            ax1.grid(True, axis='y')
            
            for bar in bars1:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.3f}', ha='center', va='bottom')
            
            
            ax2 = plt.subplot(122)
            times = []
            for norm in norms:
                norm_mask = mask & (df['norm'] == norm)
                time = df[norm_mask]['average_recognition_time'].mean()
                times.append(time)
            
            bars2 = ax2.bar(norms, times, color='lightcoral')
            ax2.set_title('Average Query Time')
            ax2.set_xlabel('Norm')
            ax2.set_ylabel('Time (s)')
            plt.xticks(rotation=45)
            ax2.grid(True, axis='y')
            
            for bar in bars2:
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.4f}', ha='center', va='bottom')
            
            plt.tight_layout()
            canvas = FigureCanvasTkAgg(fig, master=k_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(pady=5, padx=5)
            
             
            fig2 = plt.figure(figsize=(10, 4))
            
            
            ax3 = plt.subplot(121)
            for norm in norms:
                rates = []
                for val_pct in validation_pcts:
                    mask = (df['num_neighbors'] == k) & (df['norm'] == norm) & \
                           (df['validation_percent'] == val_pct)
                    rates.append(df[mask]['recognition_rate'].mean())
                ax3.plot(validation_pcts, rates, marker='o', label=norm)
            
            ax3.set_title(f'Recognition Rate vs Validation % (k={k})')
            ax3.set_xlabel('Validation %')
            ax3.set_ylabel('Rate')
            ax3.legend(loc='best', fontsize='small')
            ax3.grid(True)
            
            
            ax4 = plt.subplot(122)
            for norm in norms:
                times = []
                for val_pct in validation_pcts:
                    mask = (df['num_neighbors'] == k) & (df['norm'] == norm) & \
                           (df['validation_percent'] == val_pct)
                    times.append(df[mask]['average_recognition_time'].mean())
                ax4.plot(validation_pcts, times, marker='o', label=norm)
            
            ax4.set_title(f'Query Time vs Validation % (k={k})')
            ax4.set_xlabel('Validation %')
            ax4.set_ylabel('Time (s)')
            ax4.legend(loc='best', fontsize='small')
            ax4.grid(True)
            
            plt.tight_layout()
            canvas2 = FigureCanvasTkAgg(fig2, master=k_frame)
            canvas2.draw()
            canvas2.get_tk_widget().pack(pady=5, padx=5)
        
        
        comparison_frame = ttk.LabelFrame(parent_frame, text="k Value Comparisons")
        comparison_frame.pack(pady=5, padx=5, fill='x')
        
        fig3 = plt.figure(figsize=(10, 4))
        
        
        ax5 = plt.subplot(121)
        for norm in norms:
            rates = []
            for k in k_values:
                mask = (df['num_neighbors'] == k) & (df['norm'] == norm)
                rates.append(df[mask]['recognition_rate'].mean())
            ax5.plot(k_values, rates, marker='o', label=norm)
        
        ax5.set_title('Recognition Rate by k')
        ax5.set_xlabel('k')
        ax5.set_ylabel('Rate')
        ax5.legend(loc='best', fontsize='small')
        ax5.grid(True)
        
        ax6 = plt.subplot(122)
        for norm in norms:
            times = []
            for k in k_values:
                mask = (df['num_neighbors'] == k) & (df['norm'] == norm)
                times.append(df[mask]['average_recognition_time'].mean())
            ax6.plot(k_values, times, marker='o', label=norm)
        
        ax6.set_title('Query Time by k')
        ax6.set_xlabel('k')
        ax6.set_ylabel('Time (s)')
        ax6.legend(loc='best', fontsize='small')
        ax6.grid(True)
        
        plt.tight_layout()
        canvas3 = FigureCanvasTkAgg(fig3, master=comparison_frame)
        canvas3.draw()
        canvas3.get_tk_widget().pack(pady=5, padx=5)
        
        
        fig4 = plt.figure(figsize=(10, 4))
        
        
        ax7 = plt.subplot(121)
        manhattan_data = df[df['norm'] == 'manhattan'].pivot_table(
            values='recognition_rate',
            index='num_neighbors',
            columns='validation_percent',
            aggfunc='mean'
        )
        im1 = ax7.imshow(manhattan_data.values, aspect='auto', cmap='YlOrRd')
        plt.colorbar(im1, ax=ax7)
        ax7.set_title('Recognition Rate Heat Map\n(Manhattan norm)')
        ax7.set_xlabel('Validation %')
        ax7.set_ylabel('k')
        ax7.set_xticks(range(len(validation_pcts)))
        ax7.set_yticks(range(len(k_values)))
        ax7.set_xticklabels(validation_pcts)
        ax7.set_yticklabels(k_values)
        
        for i in range(len(k_values)):
            for j in range(len(validation_pcts)):
                text = ax7.text(j, i, f'{manhattan_data.values[i, j]:.1f}',
                              ha="center", va="center")
        
        
        ax8 = plt.subplot(122)
        manhattan_times = df[df['norm'] == 'manhattan'].pivot_table(
            values='average_recognition_time',
            index='num_neighbors',
            columns='validation_percent',
            aggfunc='mean'
        )
        im2 = ax8.imshow(manhattan_times.values, aspect='auto', cmap='YlOrRd')
        plt.colorbar(im2, ax=ax8)
        ax8.set_title('Query Time Heat Map\n(Manhattan norm)')
        ax8.set_xlabel('Validation %')
        ax8.set_ylabel('k')
        ax8.set_xticks(range(len(validation_pcts)))
        ax8.set_yticks(range(len(k_values)))
        ax8.set_xticklabels(validation_pcts)
        ax8.set_yticklabels(k_values)
        
        for i in range(len(k_values)):
            for j in range(len(validation_pcts)):
                text = ax8.text(j, i, f'{manhattan_times.values[i, j]:.4f}',
                              ha="center", va="center")
        
        plt.tight_layout()
        canvas4 = FigureCanvasTkAgg(fig4, master=comparison_frame)
        canvas4.draw()
        canvas4.get_tk_widget().pack(pady=5, padx=5)
        
        
        best_rate = df['recognition_rate'].max()
        best_rate_config = df.loc[df['recognition_rate'].idxmax()]
        fastest_config = df.loc[df['average_recognition_time'].idxmin()]
        
        stats_text = f"""Best Recognition: {best_rate:.3f}% (k={int(best_rate_config['num_neighbors'])}, Norm: {best_rate_config['norm']}, Val: {best_rate_config['validation_percent']}%)
        Fastest Config: {fastest_config['average_recognition_time']:.4f}s (k={int(fastest_config['num_neighbors'])}, Norm: {fastest_config['norm']}, Val: {fastest_config['validation_percent']}%)"""
        
        stats_label = ttk.Label(comparison_frame, text=stats_text, justify='left', font=('Courier', 9))
        stats_label.pack(pady=2, padx=5, anchor='w')
    
    def _create_projective_algorithm_plots(self, df, parent_frame, algorithm):
        """Creates visualizations for projective algorithms including preprocessing time analysis"""
        components = sorted(df['num_components'].unique())
        validation_pcts = sorted(df['validation_percent'].unique())
        norms = ['manhattan', 'euclidean', 'infinity', 'cosine']
        
        
        for comp in components:
            comp_frame = ttk.LabelFrame(parent_frame, text=f"Analysis for {comp} components")
            comp_frame.pack(pady=5, padx=5, fill='x')
            
            fig = plt.figure(figsize=(10, 4))
            
           
            ax1 = plt.subplot(121)
            mask = df['num_components'] == comp
            rates = []
            for norm in norms:
                norm_mask = mask & (df['norm'] == norm)
                rate = df[norm_mask]['recognition_rate'].mean()
                rates.append(rate)
            
            bars1 = ax1.bar(norms, rates, color='skyblue')
            ax1.set_title('Recognition Rate')
            ax1.set_xlabel('Norm')
            ax1.set_ylabel('Rate')
            plt.xticks(rotation=45)
            ax1.grid(True, axis='y')
            
            for bar in bars1:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.3f}', ha='center', va='bottom')
            
            
            ax2 = plt.subplot(122)
            times = []
            for norm in norms:
                norm_mask = mask & (df['norm'] == norm)
                time = df[norm_mask]['average_recognition_time'].mean()
                times.append(time)
            
            bars2 = ax2.bar(norms, times, color='lightcoral')
            ax2.set_title('Average Query Time')
            ax2.set_xlabel('Norm')
            ax2.set_ylabel('Time (s)')
            plt.xticks(rotation=45)
            ax2.grid(True, axis='y')
            
            for bar in bars2:
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.4f}', ha='center', va='bottom')
            
            plt.tight_layout()
            canvas = FigureCanvasTkAgg(fig, master=comp_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(pady=5, padx=5)
            
            
            fig2 = plt.figure(figsize=(10, 4))
            
           
            ax3 = plt.subplot(121)
            for norm in norms:
                times = []
                for val_pct in validation_pcts:
                    mask = (df['num_components'] == comp) & (df['norm'] == norm) & \
                           (df['validation_percent'] == val_pct)
                    times.append(df[mask]['preprocessing_time'].mean())
                ax3.plot(validation_pcts, times, marker='o', label=norm)
            
            ax3.set_title(f'Preprocessing Time vs Validation % (k={comp})')
            ax3.set_xlabel('Validation %')
            ax3.set_ylabel('Time (s)')
            ax3.legend(loc='best', fontsize='small')
            ax3.grid(True)
            
           
            ax4 = plt.subplot(122)
            for norm in norms:
                rates = []
                for val_pct in validation_pcts:
                    mask = (df['num_components'] == comp) & (df['norm'] == norm) & \
                           (df['validation_percent'] == val_pct)
                    rates.append(df[mask]['recognition_rate'].mean())
                ax4.plot(validation_pcts, rates, marker='o', label=norm)
            
            ax4.set_title(f'Recognition Rate vs Validation % (k={comp})')
            ax4.set_xlabel('Validation %')
            ax4.set_ylabel('Rate')
            ax4.legend(loc='best', fontsize='small')
            ax4.grid(True)
            
            plt.tight_layout()
            canvas2 = FigureCanvasTkAgg(fig2, master=comp_frame)
            canvas2.draw()
            canvas2.get_tk_widget().pack(pady=5, padx=5)
        
       
        comparison_frame = ttk.LabelFrame(parent_frame, text=f"{algorithm} - Overall Comparisons")
        comparison_frame.pack(pady=5, padx=5, fill='x')
        
        fig3 = plt.figure(figsize=(15, 5))
        
        ax5 = plt.subplot(131)
        for norm in norms:
            rates = []
            for comp in components:
                mask = (df['num_components'] == comp) & (df['norm'] == norm)
                rates.append(df[mask]['recognition_rate'].mean())
            ax5.plot(components, rates, marker='o', label=norm)
        
        ax5.set_title('Recognition Rate')
        ax5.set_xlabel('Components')
        ax5.set_ylabel('Rate')
        ax5.legend(loc='best', fontsize='small')
        ax5.grid(True)
        
      
        ax6 = plt.subplot(132)
        for norm in norms:
            times = []
            for comp in components:
                mask = (df['num_components'] == comp) & (df['norm'] == norm)
                times.append(df[mask]['average_recognition_time'].mean())
            ax6.plot(components, times, marker='o', label=norm)
        
        ax6.set_title('Average Query Time')
        ax6.set_xlabel('Components')
        ax6.set_ylabel('Time (s)')
        ax6.legend(loc='best', fontsize='small')
        ax6.grid(True)
        
       
        ax7 = plt.subplot(133)
        preproc_times = []
        for comp in components:
            mask = df['num_components'] == comp
            preproc_times.append(df[mask]['preprocessing_time'].mean())
        
        ax7.plot(components, preproc_times, marker='o', color='purple')
        ax7.set_title('Average Preprocessing Time')
        ax7.set_xlabel('Components')
        ax7.set_ylabel('Time (s)')
        ax7.grid(True)
        
        plt.tight_layout()
        canvas3 = FigureCanvasTkAgg(fig3, master=comparison_frame)
        canvas3.draw()
        canvas3.get_tk_widget().pack(pady=5, padx=5)
        
        
        fig4 = plt.figure(figsize=(10, 4))
        
        ax8 = plt.subplot(121)
        pivot_data = df[df['norm'] == 'manhattan'].pivot_table(
            values='recognition_rate',
            index='num_components',
            columns='validation_percent',
            aggfunc='mean'
        )
        im1 = ax8.imshow(pivot_data.values, aspect='auto', cmap='YlOrRd')
        plt.colorbar(im1, ax=ax8)
        ax8.set_title('Recognition Rate Heat Map\n(Manhattan norm)')
        ax8.set_xlabel('Validation %')
        ax8.set_ylabel('Components')
        ax8.set_xticks(range(len(validation_pcts)))
        ax8.set_yticks(range(len(components)))
        ax8.set_xticklabels(validation_pcts)
        ax8.set_yticklabels(components)
        
        for i in range(len(components)):
            for j in range(len(validation_pcts)):
                ax8.text(j, i, f'{pivot_data.values[i, j]:.1f}',
                        ha="center", va="center")
        
        
        ax9 = plt.subplot(122)
        preproc_data = df[df['norm'] == 'manhattan'].pivot_table(
            values='preprocessing_time',
            index='num_components',
            columns='validation_percent',
            aggfunc='mean'
        )
        im2 = ax9.imshow(preproc_data.values, aspect='auto', cmap='YlOrRd')
        plt.colorbar(im2, ax=ax9)
        ax9.set_title('Preprocessing Time Heat Map\n(Manhattan norm)')
        ax9.set_xlabel('Validation %')
        ax9.set_ylabel('Components')
        ax9.set_xticks(range(len(validation_pcts)))
        ax9.set_yticks(range(len(components)))
        ax9.set_xticklabels(validation_pcts)
        ax9.set_yticklabels(components)
        
        for i in range(len(components)):
            for j in range(len(validation_pcts)):
                ax9.text(j, i, f'{preproc_data.values[i, j]:.3f}',
                        ha="center", va="center")
        
        plt.tight_layout()
        canvas4 = FigureCanvasTkAgg(fig4, master=comparison_frame)
        canvas4.draw()
        canvas4.get_tk_widget().pack(pady=5, padx=5)
        
       
        best_rate = df['recognition_rate'].max()
        best_rate_config = df.loc[df['recognition_rate'].idxmax()]
        fastest_config = df.loc[df['average_recognition_time'].idxmin()]
        fastest_preproc = df.loc[df['preprocessing_time'].idxmin()]
        
        stats_text = f"""Best Recognition: {best_rate:.3f}% (Components={int(best_rate_config['num_components'])}, Norm: {best_rate_config['norm']}, Val: {best_rate_config['validation_percent']}%)
        Fastest Recognition: {fastest_config['average_recognition_time']:.4f}s (Components={int(fastest_config['num_components'])}, Norm: {fastest_config['norm']}, Val: {fastest_config['validation_percent']}%)
        Fastest Preprocessing: {fastest_preproc['preprocessing_time']:.4f}s (Components={int(fastest_preproc['num_components'])}, Val: {fastest_preproc['validation_percent']}%)"""
        
        stats_label = ttk.Label(comparison_frame, text=stats_text, justify='left', font=('Courier', 9))
        stats_label.pack(pady=2, padx=5, anchor='w')

if __name__ == "__main__":
    root = tk.Tk()
    app = FaceRecognitionUI(root)
    root.mainloop()