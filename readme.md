# **Multilingual Embedding Distillation**

This repository contains code for **distilling a large English embedding model** into a smaller, multilingual embedding model capable of handling both English and Yoruba sentences. The goal is to produce a compact model that generates embeddings for Yoruba sentences aligned with high-quality English embeddings.

## **Objective**

We aim to:

1. Start with a large pre-trained English model (`BAAI/bge-large-en-v1.5`) as a **reference model**.  
2. Train a smaller model (`BAAI/bge-base-en-v1.5`) to:  
   * Encode Yoruba sentences in a way that aligns with English embeddings.  
   * Retain embedding quality while reducing computational requirements.

This process, called **knowledge distillation**, allows the student model (smaller model) to learn from the teacher model (reference model) without needing extensive training data or resources.

---

## **How Knowledge Distillation Works**

Distillation involves transferring knowledge from a larger, more powerful model (teacher) to a smaller, more efficient model (student). This is achieved by:

1. **Reference Outputs**:

   * The teacher model generates embeddings for English sentences.  
   * These embeddings serve as a guide for training the student model.  
2. **Training the Student**:

   * The student model processes both English and Yoruba sentences.  
   * Loss functions measure how well the student’s outputs align with the teacher’s embeddings:  
     * **Mean Squared Error (MSE)** ensures that the student’s embeddings are numerically close to the teacher’s embeddings.  
     * **Kullback-Leibler Divergence (KLD)** ensures the student’s embeddings match the distribution of the teacher’s outputs.  
   * A weighted combination of these losses guides the training process.

By the end of training, the student model can generate high-quality embeddings for Yoruba and English sentences, achieving multilingual capability with reduced size and inference costs.

---

## **Running the Code**

### **Requirements**

* Python 3.8+  
* CUDA-enabled GPU (optional but recommended)  
* Dependencies: Install via `requirements.txt`

pip install \-r requirements.txt

### **Data Setup**

* Use a dataset containing **English-Yoruba sentence pairs**.  
* The dataset should be loaded into the project using `load_data` (predefined in the repository).

### **Training**

To start training, run:

python3 main.py

### **Key Parameters**

* `batch_size`: Number of samples processed at once. Default is `64`.  
* `epochs`: Number of training iterations. Default is `2`.  
* `alpha`: Weighting factor to balance MSE and KLD losses. Default is `0.5`.  
* `learning_rate`: Initial learning rate for the optimizer. Default is `2e-5`.

### **Output**

* The trained student model is saved as `model.pt` in the project directory.  
* Metrics and logs are available via **Weights and Biases (wandb)**.

Push the final model to the Hugging Face Hub using:  
 student\_model.push\_to\_hub("odunola/yoruba-embedding-model-kld")  
tokenizer.push\_to\_hub("odunola/yoruba-embedding-model-kld")

* 

---

## **Explanation of the Code**

### **Reference Model**

A large English model (`BAAI/bge-large-en-v1.5`) is loaded and enhanced with a custom **Pooler layer** to improve embedding generation. This model remains frozen during training and serves as the teacher.

### **Student Model**

The smaller student model (`BAAI/bge-base-en-v1.5`) is fine-tuned using English and Yoruba data.

### **Dataset Loader**

`load_data()` processes the dataset:

* Tokenizes English and Yoruba sentences.  
* Prepares inputs for both teacher and student models (e.g., `input_ids`, `attention_mask`).

### **Loss Functions**

1. **MSE Loss**: Matches embeddings numerically.  
2. **KLD Loss**: Matches the distribution of embeddings.  
3. **Combined Loss**: Balances the above losses with the `alpha` weighting factor.

### **Training Loop**

1. Batch inputs are processed by the teacher and student models.  
2. Losses are computed and combined.  
3. The student model is updated via backpropagation.  
4. Training progress is logged via wandb.

---

## **Final Notes**

This project demonstrates how to use knowledge distillation to adapt a monolingual model for multilingual tasks efficiently. It is a practical way to expand language support while maintaining quality and minimizing resource usage.

