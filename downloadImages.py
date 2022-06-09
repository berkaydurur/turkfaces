
from google_images_download import google_images_download 
  
# creating object
response = google_images_download.googleimagesdownload() 
  
with open("turkish_actors_and_actresses.txt", encoding='utf-8') as file:
    lines = file.readlines()
    lines = [line.rstrip() for line in lines]

  
def downloadimages(query):

    arguments = {"keywords": query or query + " face",
                 "format": "jpg",
                 "type": "face",
                 "size": ">400*300",
                 "limit":50,
                 "print_urls":True}
    try:
        response.download(arguments)
      
    # Handling File NotFound Error    
    except: 
        arguments = {"keywords": query,
                     "format": "jpg",
                     "limit":50}
                       
        # Providing arguments for the searched query
        try:
            # Downloading the photos based
            # on the given arguments
            response.download(arguments) 
        except:
            pass
  
# Driver Code
for query in lines:
    query_eng = query.replace("ç","c")
    query_eng = query_eng.replace("ğ","g")
    query_eng = query_eng.replace("ı","i")
    query_eng = query_eng.replace("ö","o")
    query_eng = query_eng.replace("ş","s")
    query_eng = query_eng.replace("ü","u")
    query_eng = query_eng.replace("Ç","C")
    query_eng = query_eng.replace("Ğ","G")
    query_eng = query_eng.replace("I","İ")
    query_eng = query_eng.replace("Ö","O")
    query_eng = query_eng.replace("Ş","S")
    query_eng = query_eng.replace("Ü","U")
    print(query_eng)
    try:
        downloadimages(query_eng)
        print(query)
        
    except Exception as e:
        print(e) 
    print()
    