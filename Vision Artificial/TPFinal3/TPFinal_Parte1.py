from utils.TPFhu_moments_generation import generate_hu_moments_file
from utils.TPFtraining_model import train_model

#guardar los momentos de mis datos de training de hu en un archivo
generate_hu_moments_file()

#entrenar el modelo
model = train_model()

# Save the trained model
model_path = './TPFinal3/TPFgenerated-files/trained_model.xml'
model.save(model_path)
print(f"Model saved to {model_path}")