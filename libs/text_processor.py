from elasticsearch import Elasticsearch 

def create_text_analyzer(es_client, filters, language):
    index_name="btg_mrs_text_analyzer"
    
    # "_english_"
    
    if es_client.indices.exists(index=index_name):
        es_client.indices.delete(index=index_name)
        
    res = es_client.indices.create(
    index=index_name,
    body={
          "settings": {
            "index": {
              "analysis": {
                "analyzer": {
                  "btg_mrs_text_analyzer": {
                    "char_filter": ["remove_accent","special_characters_filter"],  
                    "tokenizer": "whitespace",
                    "filter": filters
                  }
                },
                "filter": {
                  "synonyms": {
                    "type": "synonym",
                    "synonyms_path": "synonymous.txt",
                    "updateable": True
                  },
                  "stop_words": {
                    "type": "stop",
                    "stopwords": language,
                    "ignore_case": True
                  }
                },
                "char_filter": {
                    "remove_accent": {
                      "type": "mapping",
                      "mappings": [
                        "á => a", "ã => a", "â => a",
                        "é => e", "ê => e",
                        "í => i",
                        "ó => o", "ô => o",
                        "ú => u", "ç => c"
                      ]
                    },
                    "special_characters_filter": {
                        "pattern": "[^A-Za-z0-9]",
                        "type": "pattern_replace",
                        "replacement": " "
                    }
                }
              }
            }
          }
        },
        # Will ignore 400 errors, remove to ensure you're prompted
        ignore=400
    )
    
    if 'error' in res:
        print(res['error']['reason'])



def analyze_text_gen_synonym_aug(es_client, text=""):
    body_={
      "analyzer": "btg_mrs_text_analyzer",
      "text": text
    }

    res = es_client.indices.analyze(index="btg_mrs_text_analyzer", body=body_)
    #print(res)
    str_list = []
    str_list.append("")
    if 'tokens' in res:
        for data in res['tokens']:
           
            if data['type'].lower() == "synonym":
                str_list.append("")
                cp_str = str_list[0]
                #print("ant:", cp_str)
                rm_str =  cp_str.split(" ")[-2]
                #print("rem:", rm_str)
                str_list[-1] = cp_str.replace(rm_str, "")+data['token']+" "
            
            else:
                for i in range(0, len(str_list)):
                    str_list[i] += data['token']+" "
            
    
    return str_list
    

def analyze_text_gen_synonym_column(es_client, text=""):
    body_={
      "analyzer": "btg_mrs_text_analyzer",
      "text": text
    }

    res = es_client.indices.analyze(index="btg_mrs_text_analyzer", body=body_)
    #print(res)
    str_main = ""
    str_synonym = ""
  
    if 'tokens' in res:
        for data in res['tokens']:
           
            if data['type'].lower() == "synonym":
                str_synonym += data['token']+" "
            
            else:
                str_main += data['token']+" "
            
    
    return str_main, str_synonym
    

def analyze_text_clean(es_client, text=""):
    body_={
      "analyzer": "btg_mrs_text_analyzer",
      "text": text
    }

    res = es_client.indices.analyze(index="btg_mrs_text_analyzer", body=body_)
    #print(res)
    str_main = ""
   
    if 'tokens' in res:
        for data in res['tokens']:
           
            if data['type'].lower() == "word":
                str_main += data['token']+" "
            
    
    return str_main  


def analyze_text_get_tokens(es_client, text=""):
    body_={
      "analyzer": "btg_mrs_text_analyzer",
      "text": text
    }

    res = es_client.indices.analyze(index="btg_mrs_text_analyzer", body=body_)
  
    token_list = []
   
    if 'tokens' in res:
        for data in res['tokens']:
            if data['type'].lower() == "word":
                token_list.append(data['token'])
            
    
    return token_list 