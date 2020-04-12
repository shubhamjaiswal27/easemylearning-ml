import pickle

model = pickle.load(open('./model/ActiveVsReflective.model', 'rb'))

# # Predict Output
predicted = model.predict([[1, 0, 1]])
print("Predicted Value:", predicted)
