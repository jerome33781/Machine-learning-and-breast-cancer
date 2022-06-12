
import os


## fichier .exe

from Algo_Clustering import *
from algo_knn_ameliore import *

def execution () :

    def onclick_mass_clustering () :
        print (interface_mass_clustering())

    def onclick_mass_knn () :
        print(interface_mass_knn())

    def onclick_wis_clustering () :
        print (interface_wisconsin_clustering())

    def onclick_wis_knn () :
        print(interface_wisconsin_knn())




    def interfaces_mass() :

        # créer fenetre
        window = Tk()

        # pesonnalise fenetre
        window.title("Breast Cancer Predictor mass ")
        window.geometry("1080x720")
        window.minsize(480,360)
        window.iconbitmap("logo_predictor.ico")
        window.config(background='#F6C0DD')

        # créer frame

        frame_globale = Frame(window,bg='#F6C0DD')



        # ajout texte global
        texte_globale = Label(frame_globale, text = "Veuillez choisir votre méthode de classification",pady = 15, font = ('arial', 24) , bg = '#F6C0DD' ,fg = '#FFFFFF')
        texte_globale.pack()

            #ajouter bouton

        clustering_button = Button(frame_globale,text = 'K-means-clustering', font = ('arial', 20) , bg = '#C56D9C', fg = '#FFFFFF',command = onclick_mass_clustering)
        clustering_button.pack(pady = 20, side = BOTTOM , fill = X)

        knn_button = Button(frame_globale,text = 'K-nearest-neibhours', font = ('arial', 20) , bg = '#C56D9C', fg = '#FFFFFF',command = onclick_mass_knn)
        knn_button.pack(pady = 20, side = BOTTOM , fill = X)

        frame_globale.pack(expand = YES)

        window.mainloop()



    def interfaces_wis () :

        # créer fenetre
        window = Tk()

        # pesonnalise fenetre
        window.title("Breast Cancer Predictor wisconsin ")
        window.geometry("1080x720")
        window.minsize(480,360)
        window.iconbitmap("logo_predictor.ico")
        window.config(background='#F6C0DD')

        # créer frame

        frame_globale = Frame(window,bg='#F6C0DD')



        # ajout texte global
        texte_globale = Label(frame_globale, text = "Veuillez choisir votre méthode de classification",pady = 15, font = ('arial', 24) , bg = '#F6C0DD' ,fg = '#FFFFFF')
        texte_globale.pack()

            #ajouter bouton

        clustering_button = Button(frame_globale,text = 'K-means-clustering', font = ('arial', 20) , bg = '#C56D9C', fg = '#FFFFFF',command = onclick_wis_clustering)
        clustering_button.pack(pady = 20, side = BOTTOM , fill = X)

        knn_button = Button(frame_globale,text = 'K-nearest-neibhours', font = ('arial', 20) , bg = '#C56D9C', fg = '#FFFFFF',command = onclick_wis_knn)
        knn_button.pack(pady = 20, side = BOTTOM , fill = X)

        frame_globale.pack(expand = YES)

        window.mainloop()




    def onclick_mass_global () :
        print(interfaces_mass())

    def onclick_wis_global () :
        print(interfaces_wis())





    # créer fenetre
    window = Tk()

    # pesonnalise fenetre
    window.title("Breast Cancer Predictor ")
    window.geometry("1080x720")
    window.minsize(480,360)
    window.iconbitmap("logo_predictor.ico")
    window.config(background='#F6C0DD')

    # créer frame

    frame_globale = Frame(window,bg='#F6C0DD')



    # ajout texte global
    texte_globale = Label(frame_globale, text = "Veuillez choisir votre base de données",pady = 15, font = ('arial', 24) , bg = '#F6C0DD' ,fg = '#FFFFFF')
    texte_globale.pack()

        #ajouter bouton

    mass_button = Button(frame_globale,text = 'Mammographic mass dataset', font = ('arial', 20) , bg = '#C56D9C', fg = '#FFFFFF',command = onclick_mass_global)
    mass_button.pack(pady = 20, side = BOTTOM , fill = X)

    wis_button = Button(frame_globale,text = 'Breast cancer wisconsin dataset', font = ('arial', 20) , bg = '#C56D9C', fg = '#FFFFFF',command = onclick_wis_global)
    wis_button.pack(pady = 20, side = BOTTOM , fill = X)

    frame_globale.pack(expand = YES)

    window.mainloop()


print(execution())