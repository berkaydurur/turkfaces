from PIL import Image, ImageDraw
import face_recognition
import glob
from pathlib import Path
import os

with open("turkish_actors_and_actresses.txt", encoding='utf-8') as file:
    lines = file.readlines()
    lines = [line.rstrip() for line in lines]
#
#print(lines)
# Load the jpg file into a numpy array

def shapeFace(original_image, top, bottom, right, left):  # TODO Bu fonksiyonu gozden gecir.
    try:
        image_height = bottom - top
        image_height = int(image_height)
        image_width = right - left
        image_width = int(image_width)
        top = int(top - int(image_height * 0.15))
        bottom = int(bottom + int(image_height * 0.15))
        right = int(right + int(image_width * 0.15))  # TODO Burasi width olmali.
        left = int(left - int(image_width * 0.15))  # TODO Burasi width olmali.
        image_height = bottom - top
        image_height = int(image_height)
        image_width = right - left
        image_width = int(image_width)
        print(image_height, image_width)

        if (image_height >= 175 and image_width >= 175):
            if (image_height < 224 and image_width < 224):
                expand_value = 224 - image_height
                bottom = bottom + (expand_value / 2)
                top = top - (expand_value / 2)
                expand_value = 224 - image_width
                right = right + (expand_value / 2)
                left = left - (expand_value / 2)

            else:
                if (image_height > image_width):
                    expand_value = image_height - image_width
                    right = right + (expand_value / 2)
                    left = left - (expand_value / 2)

                    #to do buraya esitlik durumunu ekle
                else:
                    expand_value = image_width - image_height
                    bottom = bottom + (expand_value / 2)
                    top = top - (expand_value / 2)
        print(image_height, image_width)
        # print("New Values : ",top, bottom, left, right,image_height, image_width)
        top = int(top)
        bottom = int(bottom)
        right = int(right)
        left = int(left)
        image_height = bottom - top
        image_height = int(image_height)
        image_width = right - left
        image_width = int(image_width)

        draw = ImageDraw.Draw(original_image)
        draw.rectangle(((left, top), (right, bottom)), fill=None, outline="red", width=3)
        original_image.save("deneme.jpg")

        return top, bottom, left, right, image_height, image_width
    except Exception as e:
        print(e)
        return 0

def checkLimits(original_width, original_height, top, bottom, left, right):
    try:
        print("Checklimits Start", original_width, original_height, top, bottom, left, right)
        padding_top_value = 0
        padding_bottom_value = 0
        padding_right_value = 0
        padding_left_value = 0
        print()
        if (top < 0):
            padding_top_value = abs(top)
            top = 0
        #   print("TOP")
        if (bottom > original_height):
            padding_bottom_value = bottom - original_height
            bottom = original_height
        #  print("BOTTOM")
        if (left < 0):
            padding_left_value = abs(left)
            left = 0
        # print("LEFT")
        if (right > original_width):
            padding_right_value = right - original_width
            right = original_width
            print("Checklimits End", top, bottom, left, right, padding_top_value, padding_bottom_value, padding_left_value, padding_right_value)
        return top, bottom, left, right, padding_top_value, padding_bottom_value, padding_left_value, padding_right_value
    except Exception as e:
        print(e)
        return 0
def add_padding(pil_img, pad_top, pad_bottom, pad_left, pad_right):
    try:
        width, height = pil_img.size
        print("WIDTH", width + pad_right + pad_left)
        print("HEIGHT" ,height + pad_top + pad_bottom)
        new_width = width + pad_right + pad_left
        new_height = height + pad_top + pad_bottom
        if (new_width > new_height):
            new_height = new_width
        else:
            new_width = new_height
        print("Pad values", pad_left, pad_top)
        if(new_height < 224 or new_width < 224):
            new_height = 224
            new_width = 224
            original_photo_location_width = int((224 - width)/2)
            original_photo_location_height = int((224 - height)/2)
            print("original locs ",original_photo_location_width,  original_photo_location_height)
            result = Image.new(pil_img.mode, (new_width, new_height), (0))
            result.paste(pil_img, (original_photo_location_width,original_photo_location_height))
            print("Padded_image size : ",result.size)
            # print("Result size : ",result.size)   
        else:
            result = Image.new(pil_img.mode, (new_width, new_height), (0))
            result.paste(pil_img, (pad_left, pad_top))
            print("Padded_image size : ",result.size)
            # print("Result size : ",result.size)
        return result
    except Exception as e:
        print(e)
        return 0

def getFacesFromImage(query):
    
    # print("faces_final/"+ query + "_faces/")
    # Path("faces_final/"+ query + "_faces").mkdir(parents=True, exist_ok=True)
    face_counter = 0
    print(query)
    for filename in glob.glob('imagedownloader_v2/google-images-download/downloads/' + query + '/*.jpg'):  # assuming gif
        try:
            print(filename)
            original_image = Image.open(filename)
            original_width, original_height = original_image.size
            # print("Original Size : ", original_width, original_height)
            image = face_recognition.load_image_file(filename, mode="RGB")

            # Find all the faces in the image using the default HOG-based model.
            # This method is fairly accurate, but not as accurate as the CNN model and not GPU accelerated.
            # See also: find_faces_in_picture_cnn.py
            face_locations = face_recognition.face_locations(image)

            # print("I found {} face(s) in this photograph.".format(len(face_locations)))
            print(len(face_locations))
            # print(face_locations)
            if (len(face_locations) == 1):  #'''len(face_locations) == 1'''
                # print("KABUL EDILDI")
                print()
                for face_location in face_locations:
                    # Print the location of each face in this image
                    top, right, bottom, left = face_location
                    print("Old Values : ",top, bottom, right, left, bottom-top, right-left)
                    cropped_top, cropped_bottom, cropped_left, cropped_right, cropped_image_height, cropped_image_weight = shapeFace(
                        original_image, top, bottom, right, left)
                    # print("New Values : ",cropped_top, cropped_bottom, cropped_left , cropped_right, cropped_image_height, cropped_image_weight)
                    cropped_top, cropped_bottom, cropped_left, cropped_right, pad_top, pad_bottom, pad_left, pad_right = checkLimits(
                        original_width, original_height, cropped_top, cropped_bottom, cropped_left, cropped_right)
                    # print("Newest Values : ",cropped_top, cropped_bottom, cropped_left , cropped_right, cropped_bottom-cropped_top, cropped_right-cropped_left)
                    # print("Padding Values : ", pad_top, pad_bottom, pad_left, pad_right)
                    # print("A face is located at pixel location Top: {}, Left: {}, Bottom: {}, Right: {}".format(top, left, bottom, right))
                    # You can access the actual face itself like this:
                    face_image = image[cropped_top:cropped_bottom, cropped_left:cropped_right]
                    pil_image = Image.fromarray(face_image)
                    final_image = add_padding(pil_image, pad_top, pad_bottom, pad_left, pad_right)
                    # draw = ImageDraw.Draw(original_image)
                    # draw.rectangle(((left, top), (right, bottom)), fill=None, outline="red", width=3)
                    if (os.path.isdir("faces_final/" +  query.replace(" ","_") + "/") == False):
                        os.makedirs("faces_final/" + query.replace(" ","_") + "/")
                    print(face_counter)
                    final_image.save("faces_final/" + query.replace(" ","_") + "/" + query.replace(" ","_") + "_" + str(face_counter) + '.jpg')
                    #final_image.show()
                    # original_image.save("faces_final/"+ query +"/" + query + "_" + str(face_counter) + '.jpg')
                    face_counter += 1
                    
            else:
                print()
        except Exception as e:
            print(e)
            print("PROBLEMMM")

counter = 0
# print(counter)
if (os.path.isdir("faces_final") == False):
    os.makedirs("faces_final")
for query in lines:
    #if (counter == 0):
    print(query)
    query_eng = query.replace("ç", "c")
    query_eng = query_eng.replace("ğ", "g")
    query_eng = query_eng.replace("ı", "i")
    query_eng = query_eng.replace("ö", "o")
    query_eng = query_eng.replace("ş", "s")
    query_eng = query_eng.replace("ü", "u")
    query_eng = query_eng.replace("Ç", "C")
    query_eng = query_eng.replace("Ğ", "G")
    query_eng = query_eng.replace("I", "İ")
    query_eng = query_eng.replace("Ö", "O")
    query_eng = query_eng.replace("Ş", "S")
    query_eng = query_eng.replace("Ü", "U")
    getFacesFromImage(query_eng)
        #counter = counter + 1



