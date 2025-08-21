import pandas as pd
import numpy as np
import re
import string
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping
import gradio as gr
import pickle

# -------------------------
# 1. تحميل البيانات
# -------------------------
df = pd.read_csv(r"C:\Users\am\Documents\app.py\archive (2)\YoutubeCommentsDataSet.csv")
 # عدل المسار إذا لزم

# تعديل الأعمدة
df.columns = ['Comment', 'Sentiment']
df = df.dropna()

# -------------------------
# 2. تنظيف النصوص
# -------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = re.sub(r'\@\w+|\#', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    text = text.strip()
    return text

df['cleaned'] = df['Comment'].apply(clean_text)

# -------------------------
# 3. التحويل إلى تسلسل
# -------------------------
tokenizer = Tokenizer(num_words=10000, oov_token='<OOV>')
tokenizer.fit_on_texts(df['cleaned'])
sequences = tokenizer.texts_to_sequences(df['cleaned'])
padded_sequences = pad_sequences(sequences, maxlen=100, padding='post')

# -------------------------
# 4. ترميز المشاعر
# -------------------------
le = LabelEncoder()
labels = le.fit_transform(df['Sentiment'])
num_classes = len(np.unique(labels))
labels_cat = to_categorical(labels)

X_train, X_test, y_train, y_test = train_test_split(
    padded_sequences, labels_cat, test_size=0.2, random_state=42
)

# -------------------------
# 5. بناء النموذج
# -------------------------
model = Sequential([
    Embedding(input_dim=10000, output_dim=128),
    Bidirectional(LSTM(64)),
    Dropout(0.4),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(num_classes, activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# -------------------------
# 6. تدريب النموذج
# -------------------------
early_stop = EarlyStopping(monitor='val_loss', patience=2)

history = model.fit(
    X_train, y_train,
    epochs=5,
    batch_size=64,
    validation_data=(X_test, y_test),
    callbacks=[early_stop]
)

# حفظ النموذج والأدوات
model.save("sentiment_model.h5")
with open('tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)
with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(le, f)

# -------------------------
# 7. واجهة Gradio
# -------------------------
def classify_comment(comment):
    text = clean_text(comment)
    seq = tokenizer.texts_to_sequences([text])
    pad = pad_sequences(seq, maxlen=100, padding='post')
    pred = model.predict(pad)
    label = le.inverse_transform([np.argmax(pred)])
    return label[0]

interface = gr.Interface(
    fn=classify_comment,
    inputs=gr.Textbox(lines=3, placeholder="Type your YouTube comment here..."),
    outputs="text",
    title="YouTube Comment Sentiment Classifier"
)

interface.launch()
