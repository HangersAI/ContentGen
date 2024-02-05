# import os
# import uvicorn
# # import cv2
# # import numpy as np
# from tempfile import NamedTemporaryFile
# from modules.tagging import TaggingEngine
# import time
# from fastapi import  UploadFile,  FastAPI, File
# from fastapi.middleware.cors import CORSMiddleware
# from fastapi.exceptions import HTTPException
# import time

# app = FastAPI()
# origins = ["*"]

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=origins,
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# modelEngine = TaggingEngine(model_path= 'mask_rcnn_model')

# @app.post("/upload_image/")
# async def upload(input_image: UploadFile = File(...)):

#     if input_image.file is None:
#         return None
#         # return ErrorCode("Vui lòng truyền dữ liệu ảnh vào formdata")
    
#     # check the content type (MIME type)
#     if input_image.content_type not in ["image/jpeg", "image/png", "image/gif"]:
#         raise HTTPException(status_code=400, detail="Invalid file type")

#     temp = NamedTemporaryFile(delete=False, suffix=  '.' + input_image.filename.split('.')[-1])
#     contents = input_image.file.read()
#     with temp as f:
#         f.write(contents)
#     # print(temp.name)

#     start = time.time()
#     prediction = modelEngine.general_tagging(image_path = temp.name)
#     end = time.time()
#     print('Process time:', end-start)

#     os.remove(temp.name)
    
#     input_image.file.close()
#     return prediction


# @app.post("/upload_multiple_images/")
# async def upload_files(input_images: list[UploadFile]):
#     results = {}
#     for input_image in input_images:
#         if input_image.content_type not in ["image/jpeg", "image/png", "image/gif"]:
#             results[input_image.filename] = "Invalid file type"
#         else:
#             temp = NamedTemporaryFile(delete=False, suffix=  '.' + input_image.filename.split('.')[-1])
#             contents = input_image.file.read()
#             with temp as f:
#                 f.write(contents)
#             # print(temp.name)

#             start = time.time()
#             prediction = modelEngine.general_tagging(image_path = temp.name)
#             end = time.time()
#             print('Process time:', end-start)

#             os.remove(temp.name)
#             results[input_image.filename] = prediction
    
#     return results


# if __name__=="__main__":
#     uvicorn.run(app,host="0.0.0.0",port=8000)


import uvicorn
from api.app import app

if __name__ == "__main__":
    uvicorn.run(app=app, host="127.0.0.1", port=8001)
