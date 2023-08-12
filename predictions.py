import cv2
import torch
import autisum_model


model=autisum_model.make_model()
model.eval()


nrows = 150
ncolumns  = 150
channels = 3


capture=cv2.VideoCapture(0)
while True:
    istrue,image=capture.read()

    #image='D:\\Hand-Classification-For-Autism-Diagnosis-main\\Hand-Classification-For-Autism-Diagnosis-main\\faceImage\\archive\\AutismDataset\\consolidated\\Autistic\\0001.jpg'
    img=cv2.resize(image, (nrows, ncolumns), interpolation = cv2.INTER_CUBIC)
    tensor_img=torch.Tensor(img)
    tensor_img=torch.permute(tensor_img, (2,0,1))
    tensor_img=torch.unsqueeze(tensor_img,dim=0)


    preds=model(tensor_img)
    print(preds)



    if preds.item()>0.55:
        cv2.putText(image,'auistic',(50,50),cv2.FONT_HERSHEY_COMPLEX,2,(255,50,50),4)
        print('autistic')
    else:
        cv2.putText(image,'non_auistic',(50,50),cv2.FONT_HERSHEY_COMPLEX,2,(255,50,50),4)
        print('non_autistic')
    
    cv2.imshow('preds',image)
    if cv2.waitKey(20) & 0xFF==ord('d'):
        break

capture.release()
cv2.destroyAllWindows()