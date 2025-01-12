import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Add, Dropout, Bidirectional, Attention
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import os

# Used Pretrained InceptionV3 Model for Feature Extraction
base_model = InceptionV3(weights='imagenet')
model = Model(inputs=base_model.input, outputs=base_model.layers[-2].output)

# Fine-tune the last few layers of InceptionV3
for layer in base_model.layers[-10:]:
    layer.trainable = True

def extract_features(image_path):
    """Extract features from an image using InceptionV3."""
    try:
        img = load_img(image_path, target_size=(299, 299))
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        features = model.predict(img_array, verbose=0)
        return features
    except Exception as e:
        print(f"Error extracting features from {image_path}: {e}")
        return None

# Load and Preprocess Dataset
image_dir = "path/to/images"
captions_file = "path/to/captions.txt"

captions = {}
with open(captions_file, 'r') as file:
    for line in file:
        image_id, caption = line.strip().split('\t')
        image_id = image_id.split('.')[0]
        if image_id not in captions:
            captions[image_id] = []
        captions[image_id].append(caption)

# Tokenize captions
tokenizer = Tokenizer(num_words=5000, oov_token="<unk>")
all_captions = [cap for caps in captions.values() for cap in caps]
tokenizer.fit_on_texts(all_captions)
vocab_size = len(tokenizer.word_index) + 1
max_length = max(len(cap.split()) for cap in all_captions)

def preprocess_captions(captions):
    sequences = tokenizer.texts_to_sequences(captions)
    padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post')
    return padded_sequences

# Extract features for all images and store them in a dictionary
image_features = {}
for image_id in captions.keys():
    image_path = os.path.join(image_dir, image_id + '.jpg')
    features = extract_features(image_path)
    if features is not None:
        image_features[image_id] = features

# Preparing input-output pairs
X1, X2, y = [], [], []
for image_id, caps in captions.items():
    for cap in caps:
        seq = tokenizer.texts_to_sequences([cap])[0]
        for i in range(1, len(seq)):
            in_seq, out_seq = seq[:i], seq[i]
            in_seq = pad_sequences([in_seq], maxlen=max_length, padding='post')[0]
            out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
            if image_id in image_features:
                X1.append(image_features[image_id])
                X2.append(in_seq)
                y.append(out_seq)

X1, X2, y = np.array(X1), np.array(X2), np.array(y)

#Defining the Model with Bidirectional LSTM and Attention
image_input = Input(shape=(2048,))
image_dense = Dense(256, activation='relu')(image_input)

description_input = Input(shape=(max_length,))
description_embedding = Embedding(input_dim=vocab_size, output_dim=256, mask_zero=True)(description_input)
description_lstm = Bidirectional(LSTM(256))(description_embedding)

#  Adding Attention Mechanism
attention_layer = Attention()([image_dense, description_lstm])
combined = Add()([image_dense, description_lstm, attention_layer])
dropout = Dropout(0.5)(combined)
output = Dense(vocab_size, activation='softmax')(dropout)

model = Model(inputs=[image_input, description_input], outputs=output)
model.compile(optimizer='adam', loss='categorical_crossentropy')

# Callbacks for model checkpointing and early stopping
checkpoint = ModelCheckpoint("image_captioning_model.h5", save_best_only=True, monitor="loss", mode="min", verbose=1)
early_stop = EarlyStopping(monitor="loss", patience=5, restore_best_weights=True, verbose=1)

# Model training with callbacks
epochs = 20
batch_size = 128
model.fit([X1, X2], y, epochs=epochs, batch_size=batch_size, callbacks=[checkpoint, early_stop])

# Caption generation using beam search
def generate_caption(image_path, model, tokenizer, max_length, beam_width=3):
    """Generate a caption for a given image using beam search."""
    feature = extract_features(image_path)
    in_text = '<start>'
    sequences = [[list(), 1.0]]
    
    for _ in range(max_length):
        all_candidates = []
        for seq, score in sequences:
            if len(seq) > 0 and seq[-1] == tokenizer.word_index.get('<end>'):
                all_candidates.append([seq, score])
                continue
            
            sequence_input = pad_sequences([seq], maxlen=max_length, padding='post')
            yhat = model.predict([feature, sequence_input], verbose=0)
            yhat = np.argsort(yhat[0])[::-1][:beam_width]  # Beam search
            
            for word_index in yhat:
                word = tokenizer.index_word.get(word_index)
                if word is None:
                    continue
                candidate = [seq + [word_index], score * yhat[word_index]]
                all_candidates.append(candidate)
        
        # Sort all candidates by score
        ordered = sorted(all_candidates, key=lambda tup: tup[1], reverse=True)
        sequences = ordered[:beam_width]
    
    final_sequence = sequences[0][0]
    caption = ' '.join([tokenizer.index_word.get(i) for i in final_sequence if i != 0])
    return caption

# Example Usage
caption = generate_caption("path/to/test_image.jpg", model, tokenizer, max_length)
print("Generated Caption:", caption)
