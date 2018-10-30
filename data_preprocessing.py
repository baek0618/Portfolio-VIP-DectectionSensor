import face_recognition
import glob
import cv2

img_size = 64
img_path = './img_data/boss/'  #'./img_data/the_other/'


#####

from os.path import getsize
from os import remove
from PIL import Image
import sys

print('[INFO] 크기 및 비율이 비정상인 데이터를 삭제합니다.')

imglist = glob.glob(img_path+'*.png') 
cnt = 0 
cnt2 = 0
for img_file in imglist:
    if not getsize(img_file) or getsize(img_file)<=2000:     # 용량이 0일때 삭제
        cnt+=1
        remove(img_file)

    elif getsize(img_file):   # 가로/세로 또는 세로/가로 비율이 비정상 적일 경우 삭제
        img = Image.open(img_file)
        img_h, img_w = img.size[0], img.size[1]
        if img_h/img_w >= 3 or img_w/img_h >=3:
            cnt2+=1
            img.close()
            remove(img_file)
            
print('----크기가 비정상인 파일 : '+str(cnt) + '건 삭제')
print('----비정상 비율인 파일 : '+str(cnt2) + '건 삭제',end ='\n\n')



####

print('[INFO]',img_path,'의 이미지에서 얼굴을 추출합니다.')
print('----이미지에서 얼굴을 추출 중...')
imglist = glob.glob(img_path+'*.jpg')  
imglist2 = glob.glob(img_path+'*.png')  
imglist.extend(imglist2)

imgNum=0
cnt5=0
for img_file in imglist:
    cnt5+=1
    print('{}\r'.format(str(cnt5)+'/'+ str(len(imglist))+ '진행 중..'), end='')
    sys.stdout.flush()
    print(cnt5)
    img = cv2.imread(img_file)
    #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    faces = face_recognition.face_locations(rgb,model='cnn')

    for top, right, bottom, left in faces:
        cropped = img[top: bottom, left: right]
        # 이미지를 저장
        cv2.imwrite(img_path+'cvt_'+ str(imgNum) + ".png", cropped)
        imgNum += 1
    remove(img_file)

print('----'+str(imgNum) + '건의 얼굴을 추출했습니다. ',end='\n\n')
###img resize 

print('[INFO]','이미지의 사이즈를',img_size,'으로 수정합니다.')

imglist =glob.glob(img_path+"*png")

img = Image.open(imglist[0])

for img_path in imglist:
    img = Image.open(img_path)
    img.resize((img_size,img_size)).save(img_path)
	
print('[INFO] 작업이 성공적으로 수행되었습니다.')
