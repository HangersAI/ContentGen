import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig

from clip.helper import CategoryIdentifier

CATEGORIES = [
        "Food product",
        "Drinks product",
        "Fashion product",
        "Technology electronics product",
        "Houseware product",
        "Health product",
        "Beauty product",
        ]

FASHION_CATEGORIES = [
                "Clothing product",
                "Footwear product",
                "Accessory product",
                ]

CATE2CLS = {
        "Technology electronics product": "electronic",
        "Houseware product": "household",
        "Health product": "health_beauty",
        "Beauty product": "health_beauty",
        "Food product": "food",
        "Drinks product": "food",
        "Clothing product": "clothing",
        "Footwear product": "footwear",
        "Accessory product": "accessory",
    }


class TaggingEngine:
    def __init__(self, model_path) :
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_path, 
                                                    device_map="cuda", trust_remote_code=True).eval()
        self.original_config = GenerationConfig.from_pretrained(model_path, 
                                                                trust_remote_code=True)
        self.model.generation_config = self.original_config

        self.category_identifier = CategoryIdentifier(self.device)
    
    def set_config(self, top_k = 1, top_p = 0.8):
        # set up new config
        self.model.generation_config.top_k = top_k
        self.model.generation_config.top_p = top_p
    
    def reset_config(self):
        # reset to original config
        self.model.generation_config = self.original_config
    
    def classify(self, image_path):
        category_pred = self.category_identifier(image_path, CATEGORIES, FASHION_CATEGORIES)
        category_pred = CATE2CLS[category_pred]
        return category_pred
    
    def postprocess_tagging(self, response):
        def remove_none_tagging(response):
            # loại bỏ các thuộc tính None
            new_response = {key: value for key, value in response.items() if value != "None" and key != "location"}
            return new_response
        
        # postprocess
        try:
            response = json.loads(response)
            response = remove_none_tagging(response)
            response = {
                        "message": "Success",
                        "status_code":200,
                        "infos": response
                        }
        except:
            print('Try replacing to parse json')
            try:
                if response.count("}") == 0:
                    response = {
                        "message": "Cannot parse json format",
                        "status_code":400,
                        "infos": {}
                        }
                elif response.count("}") == 1:
                    response = response[:response.index("}") + 1]
                    response = json.loads(response.replace('\n', '').replace(',}', '}'))
                    response = remove_none_tagging(response)
                    response = {
                            "message": "Success",
                            "status_code":200,
                            "infos": response
                            }
                else:
                    response = response[:response.index("}") + 1]
                    response = json.loads(response.replace('\n', '').replace(',}', '}'))
                    response = remove_none_tagging(response)
                    response = {
                            "message": "Success",
                            "status_code":200,
                            "infos": response
                            }

            except:
                print('Cannot parse json format')
                print(response)
                response = {
                        "message": "Cannot parse json format",
                        "status_code":400,
                        "infos": {}
                        }   
        return response
    
    def general_tagging(self, image_path):
        cls_pred = self.classify(image_path)

        if cls_pred == "electronic":
            return self.electronic_tagging(image_path)
        if cls_pred == "household":
            return self.household_tagging(image_path)
        if cls_pred == "health_beauty":
            return self.health_beauty_tagging(image_path)
        if cls_pred == "food":
            return self.health_beauty_tagging(image_path)
        if cls_pred in ["clothing", "footwear", "accessory"]:
            return self.fashion_tagging(image_path, fashion_type=cls_pred)
        raise

    def electronic_tagging(self, image_path):
        # set up new config
        self.set_config()

        # main prompt
        main_prompt = 'Given an image of a electronics item, provide information for the main object in a JSON format with the following properties: {"category":"Type of electronic item","brand":"Brand or origin","model":"Model name or number","features":"Key features or specifications", "color":"Color","size":"Dimensions","weight":"Weight","material":"Primary material","power_source":"Power source","connectivity":"Connectivity options","price":"Price range","special_features":"Special features or capabilities","warranty":"Warranty period","compatibility":"Device compatibility","energy_efficiency":"Energy efficiency rating","packaging":"Packaging details","included_accessories":"Included accessories","usage_instructions": "Usage instructions or guidelines"}. If any attribute is absent or undefined, return "None". If the image is not clear, return "Invalid input".'

        query = self.tokenizer.from_list_format([
                    {'image': image_path}, # Either a local path or an url
                    {'text': main_prompt},
                ])
        # process
        response, _ = self.model.chat(self.tokenizer, query=query, history=None)
        print(response)

        # postprocess
        response = self.postprocess_tagging(response)

        # return old config
        self.reset_config() 

        return response        
    
    def household_tagging(self, image_path):
        # set up new config
        self.set_config()

        # main prompt
        main_prompt = 'Given a household items image, provide information for main object in a JSON format with the following properties: {"category": "Type of product", "shape": "Form or style", "color": "one to three primary color", "pattern": "Pattern or design", "material": "Primary material", "power": "Power source", "capacity": "Volume or capacity", "smart features": "Smart features", "brand": "Manufacturer or brand", "price": "Price range", "Special features": "Special features", "warranty": "Warranty period", "size": "Dimensions", "weight": "Weight", "assembly": "Assembly requirements". If any attribute is absent or undefined, return "None". If the image is not clear, return "Invalid input".'

        query = self.tokenizer.from_list_format([
                    {'image': image_path}, # Either a local path or an url
                    {'text': main_prompt},
                ])
        # process
        response, _ = self.model.chat(self.tokenizer, query=query, history=None)
        print(response)

        # postprocess
        response = self.postprocess_tagging(response)

        # return old config
        self.reset_config()  

        return response            

    def health_beauty_tagging(self, image_path):
        # set up new config
        self.set_config()

        # main prompt
        main_prompt = 'Given an image of a health and beauty product, provide information for the main object in a JSON format with the following properties:{"category":"Type of product(e.g:Medicinal Product,Cosmetic,Pharmaceuticals,Fragrances,skincare product)","brand":"Brand or origin","model":"Model name or number", "ingredients": "List of ingredients", "purpose": "Intended use or benefits", "recommended_usage": "Recommended usage instructions", "scent": "Fragrance or scent", "nutritional_info": "Nutritional information", "temperature": "Recommended temperature or storage conditions", "price": "Price range", "special_features": "Special features or preparation method", "expiration_date": "Expiration date", "size": "Dimensions or volume","weight":"Weight","packaging":"Packaging detail","allergens": "Allergen information"}.If any attribute is absent or undefined, return "None". If the image is not clear, return "Invalid input".'
        

        query = self.tokenizer.from_list_format([
                    {'image': image_path}, # Either a local path or an url
                    {'text': main_prompt},
                ])
        # process
        response, _ = self.model.chat(self.tokenizer, query=query, history=None)
        print(response)

        # postprocess
        response = self.postprocess_tagging(response)

        # return old config
        self.reset_config()

        return response 

    def food_tagging(self, image_path):
        # set up new config
        self.set_config()

        # main prompt
        main_prompt = 'Given a food or drink image, provide information for main object in a JSON format with the following properties: {"category": "Type of food or drink", "flavor": "taste or flavor","ingredients": "List of ingredients", "cuisine": "Culinary origin or style", "temperature": "Serving temperature", "spiciness": "Spiciness level", "sweetness": "Sweetness level", "serving size": "Portion or serving size", "nutritional info": "Nutritional information", "brand": "Brand or origin", "price": "Price range", "special features": "Special features or preparation method", "expiration_date":"Expiration date","alcohol_content": "Alcohol content for drinks","size": "Dimensions or volume", "weight": "Weight","packaging":"Packaging details","allergens":"Allergen information". If any attribute is absent or undefined, return "None". If the image is not clear, return "Invalid input".'
        

        query = self.tokenizer.from_list_format([
                    {'image': image_path}, # Either a local path or an url
                    {'text': main_prompt},
                ])
        # process
        response, _ = self.model.chat(self.tokenizer, query=query, history=None)
        print(response)

        # postprocess
        response = self.postprocess_tagging(response)

        # return old config
        self.reset_config() 
              
        return response 

    def fashion_tagging(self, image_path, fashion_type = 'clothing'):
        # set up new config
        self.model.generation_config.top_k = 1
        self.model.generation_config.top_p = 0.8

        # main prompt
        clothing_prompt = '''
                Given an image containing the clothing, extract and provide detailed information about the clothing in JSON format with following properties: \
                {"category": "Type of the clothing (e.g., shirt, blouse, pant, skirt)", "length": "The length of the clothing (e.g., short, long, regular)", "collar": "The collar shape of the clothing (e.g., round, square)", "sleeve": "The length of the sleeve in the clothing. Choose one in {short sleeve, long sleeve, no sleeve}", "waistline": "The waistline of the clothing. Choose one in {high waistline, normal waistline, low waistline, no waistline}", "color": "The color of the clothing", "pattern": "The pattern in the clothing (e.g., solid, plaid, number, character)", "material": "The material of the clothing", "opening type": "The opening type of the clothing (e.g., double breasted, single breasted, zipper)", "occasion": "The occasion we can wear the clothing (e.g., casual, formal, winter)", "location": "the location in the body we can wear the clothing. Choose one in {upper body, lower body}".  If any attribute is absent or undefined, return "None".}
                Example:
                {
                "type": "shirt",
                "length": "regular",
                "collar": "round",
                "sleeve": "short sleeve",
                "waistline": "None",
                "color": "blue",
                "pattern": "plaid pattern",
                "material": "cotton",
                "opening type": "None",
                "occasion": "casual",
                "location": "upper body"
                }
                '''
        footwear_prompt = '''
                Given an image containing the footwear, extract and provide detailed information about the footwear in JSON format with following properties: \
                {"type": "The type of footwear (e.g., shoe, sneaker, boot, sandal)", "brand": "The brand of the footwear", "model": "The specific model or name of the footwear", "color": "The color of the footwear", "material": "The material of the footwear (e.g., leather, canvas)", "lock type": "The type of closure or locking mechanism used in the footwear (e.g., cord lock)", "collar height": "The height of the collar if applicable. Choose one in {low collar, high collar, None}", "occasion": "The occasion we can wear the footwear (e.g., formal, winter, sports)"}
                If any attribute is absent or undefined, return "None". Ensure that the JSON format is valid.
                Example:
                {
                "type": "sneaker",
                "brand": "Nike",
                "model": "None",
                "color": "neon",
                "material": "mesh",
                "lock type": "cord lock",
                "collar height": "low collar",
                "occasion": "sports"
                }
                '''
        accessory_prompt =  '''
                Given an image containing the accessory, extract and provide detailed information about the accessory in JSON format with the following properties:
                {"type": "The type of accessory (e.g., bag, necklace, etc.)", "brand": "The brand of the accessory", "model": "The specific model or name of the accessory", "color": "The main color of the accessory (e.g., white, black, etc.)", "material": "The material of the accessory (e.g., leather, canvas, etc.)", "pattern": "The patterns on the accessory (e.g., plaid, flower, etc.)", "lock type": "The type of closure or locking mechanism used in the accessory", "occasion": "The occasion we can use the accessory (e.g., formal, casual, party, etc.)"}
                If any attribute is absent or undefined, return "None". Ensure that the JSON format is valid.
                Example:
                {
                "type": "bag",
                "brand": "None",
                "model": "None",
                "color": "white",
                "material": "canvas",
                "pattern": "flowers",
                "lock type": "None",
                "occasion": "Casual, outdoor activities"
                }
                '''
        
        # choose fashion type
        if fashion_type == 'clothing':
            main_prompt = clothing_prompt
        elif fashion_type == 'footwear':
            main_prompt = footwear_prompt
        elif fashion_type == 'accessory':
            main_prompt = accessory_prompt
        else:
            print('No support type')

        query = self.tokenizer.from_list_format([
                    {'image': image_path}, # Either a local path or an url
                    {'text': main_prompt},
                ])
        # process
        response, _ = self.model.chat(self.tokenizer, query=query, history=None)
        print(response)

        # postprocess
        response = self.postprocess_tagging(response)

        # return old config
        self.reset_config() 
        
        return response