from utils.hu_moments_generation import generate_hu_moments_file
from utils.training_model import train_model

#guardar los momentos de mis datos de training de hu en un archivo
generate_hu_moments_file()

#entrenar el modelo
model = train_model()

# Save the trained model
model_path = './TP2/generated-files/trained_model.xml'
model.save(model_path)
print(f"Model saved to {model_path}")




