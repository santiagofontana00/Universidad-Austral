from utils.hu_moments_generation import generate_hu_moments_file
from utils.testing_model import load_and_test
from utils.training_model import train_model

generate_hu_moments_file() 
model = train_model()
load_and_test(model)


#hay que guardar el modelo en un archivo en vez de reentrenarlo cada vez
#hay que ver que pasa si en el archivo de entrenamiento hay mas de una imagen

#una rchivo genera el csv
#ebtrenamiento?
#el tercero es el de match shapes(?. Esto quiere decir que hay que ponerlo a andar como el tp1, para un observador ajeno al tema se va  a ver igual. PEro nosotros sabemos que es distinto, por ejemplo, no hay unknowns en este caso, y en el tp1 si.

#tengo que pasar las cosas a escala logaritmica


#el csv no tiene que ser con nombres sino con labels, no tiene que ser estrella, cuadrado,etc sino 0,1,2,3,etc



