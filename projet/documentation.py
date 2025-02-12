import os
def readDocumentation():
    # Nom du fichier à lancer
    fichier_a_lancer = "systeme prediction.pdf" # Remplacez par le nom de votre fichier

    # Chemin absolu du fichier
    chemin_fichier = os.path.abspath(fichier_a_lancer)

    # Lancement du fichier
    os.startfile(chemin_fichier) 
