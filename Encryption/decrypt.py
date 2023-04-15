
def rotate(front,back,top,bottom,left,right,itr,i,c):
    import numpy as  np
    sz=front.shape[0]
    
    if itr == 1:
        front,back,top,bottom,left,right = right,left,np.rot90(top,3),np.rot90(bottom,1),front,back
    elif itr == 2 :
        front,back,top,bottom,left,right = top,np.rot90(bottom,2),np.rot90(back,2),front,np.rot90(left,3),np.rot90(right,1)


    if c == 0:
        temp = np.copy(front[i])
        front[i]= right[i]
        right[i]=back[i]
        back[i]=left[i]
        left[i]=temp
        if i == 0:
            top = np.rot90(top,3)
        elif i == sz-1:
            bottom = np.rot90(bottom,1)
    elif c == 1:
        temp = np.copy(front[i])
        front[i]= left[i]
        left[i]=back[i]
        back[i]=right[i]
        right[i]=temp
        if i == 0:
            top = np.rot90(top,1)
        elif i == sz-1:
            bottom = np.rot90(bottom,3)
    elif c == 2:
        temp = np.copy(front[:,i])
        front[:,i]= bottom[:,i]
        bottom[:,i]=np.flip(back[:,sz-i-1])
        back[:,sz-i-1]=np.flip(top[:,i])
        top[:,i]=temp
        if i == 0:
            left = np.rot90(left,1)
        elif i == sz-1:
            right = np.rot90(right,3)
    else:
        temp = np.copy(front[:,i])
        front[:,i]= top[:,i]
        top[:,i]=np.flip(back[:,sz-i-1])
        back[:,sz-i-1]=np.flip(bottom[:,i])
        bottom[:,i]=temp
        if i == 0:
            left = np.rot90(left,3)
        elif i == sz-1:
            right = np.rot90(right,1)

    if itr == 1:
        temp  = np.copy(front)
        front = left
        left = back
        back = right
        top = np.rot90(top,1)
        bottom = np.rot90(bottom,3)
        right = temp
    elif itr == 2:
        temp = np.copy(front)
        front = bottom
        bottom = np.rot90(back,2)
        back = np.rot90(top,2)
        top = temp
        left = np.rot90(left,1)
        right = np.rot90(right,3)

    return front,back,top,bottom,left,right

def logistic_key(x, r, size):
    key = []
    for i in range(size):   
        x = r*x*(1-x)   
        key.append(int((x*pow(10, 16))%256))    
    return key

def decrypt():
    from tkinter import Tk     
    from tkinter.filedialog import askopenfilename
    import numpy as np
    from PIL import Image
    Tk().withdraw() 
    path = askopenfilename()
    img = Image.open(path)
    img = np.array(img)

    finalencimage  = np.copy(img)
    shrya = np.copy(img)
    for ii in range(3): 
        wholeimg  = np.copy(shrya[:,:,ii])
        img = np.copy(wholeimg)

        front = np.zeros(img.shape)
        back = np.zeros(img.shape)
        top = np.zeros(img.shape)
        bottom = np.zeros(img.shape)
        left = np.zeros(img.shape)
        right = np.zeros(img.shape)
        
        key = logistic_key(0.01, 3.85, img.shape[0]*img.shape[1])
        l =0
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                img[i,j] ^= key[l]
                pix = [int(k) for k in list('{0:08b}'.format(img[i,j]))]
                front[i][j] = pix[0]
                back[i][j]=pix[1]
                top[i][j]=pix[2]
                bottom[i][j]=pix[3]
                left[i][j]=pix[4]
                right[i][j]=pix[5]
                l+=1

        inv = {
            1:0,
            2:3,
            3:2,
            0:1
        }

        l=img.shape[0]*3-1
        for  i  in reversed(range(3)):
            for j in reversed(range(front.shape[0])):
                front,back,top,bottom,left,right = rotate(front,back,top,bottom,left,right,i,j, inv[key[l]%4])
                l-=1

        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                img[i][j] = 0
                pix = [int(k) for k in list('{0:08b}'.format(wholeimg[i,j]))]
                img[i][j] += (np.uint64(front[i][j])<<np.uint64(7))
                img[i][j] += (np.uint64(back[i][j])<<np.uint64(6))
                img[i][j] += (np.uint64(top[i][j])<<np.uint64(5))
                img[i][j] += (np.uint64(bottom[i][j])<<np.uint64(4))
                img[i][j] += (np.uint64(left[i][j])<<np.uint64(3))
                img[i][j] += (np.uint64(right[i][j])<<np.uint64(2))
                img[i][j] += (pix[6]<<1)
                img[i][j] += (pix[7])


        for x in range(1024):
            for y in range(1024):
                finalencimage[x][y][ii] = img[x][y]

    finalencimage = Image.fromarray(finalencimage)
    finalencimage.save(path)

if __name__=='__main__':
    decrypt()