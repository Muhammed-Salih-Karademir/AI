"""Music gereration firth Keras and TensorFlow backend¶***
Plan was simple:

Read midi file, convert it to matrix of features
Create simple model with Keras and LSTM to learn the pattern
Use subsample of initial midi file as a input for model to generate pure art
Save prediction from model to midi file . . .
PROFIT
For disclamer: I've been using my old Dell Laptop with no GPU support
"""
import mido
from mido import MidiFile, MidiTrack, Message
from tensorflow.python.keras.layers import LSTM, Dense, Activation, Dropout, Flatten
from tensorflow.python.keras.preprocessing import sequence
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.callbacks import ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler
import numpy as np

#Read midi file
mid = MidiFile('Samples/Klasik-Muzik-Vivaldi-Four-Seasons.mp3.mid')
notes = []

#Extract notes sequence
notes = []
for msg in mid:
    if not msg.is_meta and msg.channel == 0 and msg.type == 'note_on':
        data = msg.bytes()
        notes.append(data[1])

#Apply min-max scalling
scaler = MinMaxScaler(feature_range=(0,1))
scaler.fit(np.array(notes).reshape(-1,1))
notes = list(scaler.transform(np.array(notes).reshape(-1,1)))

#Prepare features for training and data subsample for prediction
# LSTM layers requires that data must have a certain shape
# create list of lists fist
notes = [list(note) for note in notes]
# subsample data for training and prediction
X = []
y = []
# number of notes in a batch
n_prev = 30
for i in range(len(notes)-n_prev):
    X.append(notes[i:i+n_prev])
    y.append(notes[i+n_prev])
# save a seed to do prediction later
X_test = X[-300:]
X = X[:-300]
y = y[:-300]

#Made sequential model with several layers, use LSTM as it time dependent data
#I also whant to save checkpoints

model = Sequential()
model.add(LSTM(256, input_shape=(n_prev, 1), return_sequences=True))
model.add(Dropout(0.6))
model.add(LSTM(128, input_shape=(n_prev, 1), return_sequences=True))
model.add(Dropout(0.6))
model.add(LSTM(128, input_shape=(n_prev, 1), return_sequences=True))
model.add(Dropout(0.6))
model.add(LSTM(128, input_shape=(n_prev, 1), return_sequences=True))
model.add(Dropout(0.6))
model.add(LSTM(64, input_shape=(n_prev, 1), return_sequences=False))
model.add(Dropout(0.6))
model.add(Dense(1))
model.add(Activation('tanh'))
optimizer = Adam(lr=0.001)
model.compile(loss='mse', optimizer=optimizer)
filepath="./Checkpoints/checkpoint_model_{epoch:02d}.hdf5"
model_save_callback = ModelCheckpoint(filepath, monitor='val_acc',
                                      verbose=1, save_best_only=False,
                                      mode='auto', period=5)

#Train your model.It might take a while, I was waiting for 1 hour with just 5 epoch
model.fit(np.array(X), np.array(y), 32, 160, verbose=1, callbacks=[model_save_callback])

#Make a prediction
prediction = model.predict(np.array(X_test))
prediction = np.squeeze(prediction)
prediction = np.squeeze(scaler.inverse_transform(prediction.reshape(-1,1)))
prediction = [int(i) for i in prediction]

#Save your result to new midi file
mid = MidiFile()
track = MidiTrack()
t = 0
for note in prediction:
    # 147 means note_on
    # 67 is velosity
    note = np.asarray([147, note, 67])
    bytes = note.astype(int)
    msg = Message.from_bytes(bytes[0:3])
    t += 1
    msg.time = t
    track.append(msg)
mid.tracks.append(track)
mid.save('LSTM_music_tesla_tanh_vivaldi_zabahakadar.mid')
