import os
file_dir = 'data_preprocessing/HSI'
change_dir = '...'
f1 = open('train_AVIRIS.txt','w')

for root, sub_folders, files in os.walk(file_dir):
    for name in files:
        f1.write( os.path.join(root, name) +'\n' )
        #f1.write( change_dir + os.path.join( name) +'\n' )

        
