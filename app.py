import os
import visualization

models = ['./models/{0}'.format(model) for model in os.listdir("models")]
default_text = "Arrêté ministériel N° 002 /CAB/VPM/MIN-ECONAT/DMS/TNM/2024 du 02 octobre 2024 portant fixation des prix des carburants terrestres"
visualization.visualize(models, default_text, visualizers=['ner'])
