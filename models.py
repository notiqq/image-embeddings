import torch
from transformers import AutoImageProcessor, AutoModel
from PIL import Image
import faiss
import numpy as np
import os
import core
import pickle

class SimilaritySearch():
    def __init__(self) -> None:
        self.device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
        self.processor = AutoImageProcessor.from_pretrained('facebook/dinov2-small')
        self.model = AutoModel.from_pretrained('facebook/dinov2-small').to(self.device)
        self.index = faiss.IndexFlatL2(384)
        self.images_list = None
        self.__load_related_data()
        if self.images_list == None:
            self.images_list = []

    def __add_vector_to_index(self, embedding, index):
        vector = embedding.detach().cpu().numpy()
        vector = np.float32(vector)
        faiss.normalize_L2(vector)
        index.add(vector)

    def train(self, directory, image_paths):
        for image_path in image_paths:
            try:
                img = Image.open(core.build_path(directory, image_path)).convert('RGB')
                with torch.no_grad():
                    inputs = self.processor(images=img, return_tensors="pt").to(self.device)
                    outputs = self.model(**inputs)
                features = outputs.last_hidden_state
                self.__add_vector_to_index(features.mean(dim=1), self.index)
            except:
                break
        
        self.images_list = image_paths
        faiss.write_index(self.index,"vector.index")
        self.__save_related_data()

    def search(self, directory, image_path, top_n=1):
        image = Image.open(core.build_path(directory, image_path)).convert('RGB')
        with torch.no_grad():
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            outputs = self.model(**inputs)

        embeddings = outputs.last_hidden_state
        embeddings = embeddings.mean(dim=1)
        vector = embeddings.detach().cpu().numpy()
        vector = np.float32(vector)
        faiss.normalize_L2(vector)

        index = faiss.read_index("vector.index")
        result_distances,result_indexes = index.search(vector,top_n)
        output = []
        print(result_distances,result_indexes)
        for i, index_result in enumerate(result_indexes[0]):
            output.append((result_distances[0][i], self.images_list[index_result]))
        return output
    

    def __save_related_data(self):
        try:
            with open('faiss_related_data.pkl', 'wb') as f:
                pickle.dump(self.images_list, f)
        except:
            pass

    
    def __load_related_data(self):
        try:
            with open('faiss_related_data.pkl', 'rb') as f:
                self.images_list = pickle.load(f)
        except:
            pass
        