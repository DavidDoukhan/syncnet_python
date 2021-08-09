import  pickle, os

def file_creation():
    evalCsvSave ='test.txt'
    # if (path.exists(evalCsvSave)):
    #  os.remove(evalCsvSave)
    open(evalCsvSave, 'w').close()

def hydrate_pckl_files():

    with open('../TMP2/faces.pckl', 'rb') as file:
        faces = pickle.load(file, encoding='latin1')

    with open('../TMP2/scene.pckl', 'rb') as file:
        scene = pickle.load(file, encoding='latin1')

    with open('../TMP2/tracks.pckl', 'rb') as file:
        tracks = pickle.load(file, encoding='latin1')

    with open('../TMP2/activesd.pckl', 'rb') as fil:
        activesd = pickle.load(fil, encoding='latin1')

    print('kllk')

if __name__ == '__main__':
    hydrate_pckl_files()