import cv2
from keras.models import load_model
import numpy as np

model = load_model("best_model.h5")

Number_to_emotion = {0 :'Angry', 1 : 'Disgust', 2:'Fear', 3 :'Happy', 4:'Sad', 5:'Surprise', 6:'Neutral'}
Angry = ["Basket Case","Don't Look Back in Anger","Look Back in Anger","Misery Business","Death on Two Legs","Since U Been Gone","St. Anger","Break Stuff","I Don't Care Anymore","I Hate Everything About You","Platypus (I Hate You)","Having a Blast","Faint","One Step Closer","Killing in the Name","Angry Chair","We're Not Gonna Take It","Gives You Hell","The Resistance","Prison Song","Bad Habit","It's not me It's you","Better Than Revenge","Angry Again","Kill You"]
Disgust = ["First Day Of School","I'm Not The Only One","Problem","Before He Cheats","Jar Of Hearts","Pumped Up Kicks","Jealous","Keeps Gettin' Better","SuperGirl","Since U Been Gone","That's What You Get","Call Me When You're Sober","Honey", "This Mirror Isn't Big Enough for the Two of Us","Reject","Unhappy Birthday-2011 Remaster","yes girl","Cold Shoulder","Boneyards","Pretty On The Outside","Still Don't Seem To Care","I Hate Everything About You","Throw Yourself Away","Jealous(I Ain't With It)","Really Don't Care","No Scrubs"]
Fear = ["In My Blood","Breathin","Sober","Save Myself","Perfect","Waving Through a Window","Save My Soul","Fake Happy","Head Above Water","Heavy","Fix You","The Village","Human","Who You Are","Diana","Praying","Let Your Tears Fall","Breathe Me","Smoke & Mirrors","Hard Times","The Doctor Said","Skyscraper","The Lonely","How to Save a Life","Nobody's Home"]
Happy = ["Let’s Go Crazy","Walking on Sunshine","Tightrope","Three Little Birds","Lovely Day","I Got You (I Feel Good)","Valerie","Dancing Queen","Uptown Funk","Can’t Stop the Feeling","Good as Hell","Don’t Bring Me Down","One More Time","It’s a Beautiful Morning","You Make My Dreams","Roobaroo","Ilahi","Rock On!","Yun Hi Chala Chal Rahi","Badtameez Dil","Tum Hi Ho Bandhu","Balam Pichkari","Mein Koi Aisa Geet Gaoon","Chala Jata Hoon","Dil Dhadakne Do"]
Sad = ['When We Were Young','Beyond',"I'm On Fire",'Somebody Else','Homesick','How Will I Know','From the Dining Table',"Nothing's Gonna Hurt You Baby",'ILYSB','Right Now','Wonderful Tonight',"River","Dancing on My Own","Slow Burn","Be Alright","Kabira","Agar Tum Saath Ho","Mana Ke Hum Yaar Nahi","Channa Mereya","Phir Le Aaya Dil","Main Rahoon Ya Na Rahoon","Humdard","Bulleya","Tujhe Bhula Diya","Yeh Dooriyan"]
Surprise = ["My Way","With a Little Help From My Friends","Doesn’t Make It Alright","Jealous Guy","Money (That’s What I Want)","The Sound of Silence","Comfortably Numb","Hurt","A Hard Rains a-Gonna Fall","Satisfaction","Georgia On My Mind","House of the Rising Sun","Light My Fire","War","My Sweet Lord","Family Affair","Theme from Shaft","When Doves Cry","Money For Nothing","A Whole New World","Candle in the Wind","Mo Money Mo Problems","Fallin’","Slow Jamz","Rolling in the Deep"]
Neutral = ["See You Again","Sorry","Uptown Funk","Blank Space","Shake It Off","Despacito","Lean On","Hello","Roar","Sugar","All About That Bass","Baby","Love Me Like You Do","Waka Waka","Summer Of 69","Lakshya","Kandhon Se Milte Hain Kandhe","BESABRIYAAN","Aaj Kal Zindagi","Kun Faya Kun","Salaam India","Chak De India","Badal Pe Paon Hain","All Izz Well","Chak Lein De"]
Emotion_Based_Songs = {"Angry":Angry,"Disgust":Disgust,"Fear":Fear,"Happy":Happy,"Sad":Sad,"Surprise":Surprise,"Neutral":Neutral}
cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

while True:
    res,frame = cap.read()
    
    if res==False:
        continue
    
    gray_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    
    faces = face_cascade.detectMultiScale(frame,1.2,5)
    
    if len(faces)==0:
        for (x,y,w,h) in faces:
            cv2.putText(frame,"Processing",(x,y-10),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,0),2,cv2.LINE_AA)
            
            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
    for face in faces:
        x,y,w,h = face
        
        offset = 10
        face_section = gray_frame[y-offset:y+h+offset,x-offset:x+w+offset]
        
        if (np.all(np.array(face_section.shape))):
            face_section = cv2.resize(face_section,(48,48))
            
            pred = np.argmax(model.predict(face_section.reshape(1,48,48,1)))
            label = Number_to_emotion[pred]
            SongList = Emotion_Based_Songs[label]
            Song = np.random.choice(SongList,1, p=(0.05, 0.05, 0.05, 0.05, 0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.03,0.02,0.01,0.01,0.01,0.01,0.01))[0]
            cv2.putText(frame, "Emotion: "+label, (x,y-10), cv2.FONT_HERSHEY_COMPLEX,1,(0,0,0),2,cv2.LINE_AA)
            cv2.putText(frame, "Song: "+Song, (x,y+w+30), cv2.FONT_HERSHEY_COMPLEX,1,(0,0,0),2,cv2.LINE_AA)
            cv2.rectangle(frame,(x,y),(x+w,y+h), (0,255,255),2)
            
    cv2.imshow("Emotion",frame)
    
    key_pressed = cv2.waitKey(1)
    
    if key_pressed == ord("q"):
        break
        
cap.release()
cv2.destroyAllWindows()