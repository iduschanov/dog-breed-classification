import requests

resp = requests.post("http://localhost:5000/predict",
                     files={"file": open('ILSVRC2012_val_00000590.JPEG','rb')})


print(resp.json())
# print(open('ILSVRC2012_val_00002701.JPEG','rb'))