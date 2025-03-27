import pprint
import serpscrap
import  re
keywords = ['Saeedeh Shekarpour']
snippet_corpus=''
urls=''
meta_title=''
meta_titles=''
raw_text=''
raw_text_corpus=''
url=''
snippet=''
config = serpscrap.Config( )
config_new = {
    'cachedir': '/tmp/.serpscrap/',
    'clean_cache_after': 1,
    'database_name': '/tmp/serpscrap',
    'do_caching': True,
    'num_pages_for_keyword': 5,
    'scrape_urls': True,
    'search_engines': ['google'],
    'google_search_url': 'https://www.google.com/search?',
    # 'executable_path', '/users/local/bin/chromedriver'
}
config.apply(config_new)

scrap = serpscrap.SerpScrap()
scrap.init(config=config.get(), keywords=keywords)

reg_ex_pattern=r'\'url\':(.+)\'encoding\'|\'serp_snippet\':(.+)\'serp_sitelinks\'|\'meta_title\':(.+)\'status\'|\'text_raw\':(.+)}'


results = scrap.run()

for result in results:
   web_download = re.findall(reg_ex_pattern, str(result), re.DOTALL)
    # web_download = reg_ex_pattern.findall(str(result))
   try:
     snippet = web_download[0][1]
   except:
       pass
   try:
     url = web_download[2][0]
   except:
        pass
   try:
      meta_title = web_download[1][2]
   except:
       pass
   try:
      raw_text = web_download[3][3]
   except:
       pass
   snippet_corpus += snippet[2:-3] + '\n\n'
   urls += url[2:-3] + '\n'
   raw_text_corpus+=raw_text[2:-3]+'\n'
   meta_titles+=meta_title[2:-3]+'\n'
snippet_corpus=snippet_corpus.replace('\\xa0','')
snippet_corpus=snippet_corpus.replace('...','')
snippet_corpus=snippet_corpus.replace('\"','')
snippet_corpus=snippet_corpus.replace('-','')
raw_text_corpus=raw_text_corpus.replace('\"','')
print(urls)
print()
print(snippet_corpus)
print()
# print('======================================================================================================================')
# print()
print(raw_text_corpus)
# print()
# print('======================================================================================================================')
print()
print(meta_titles)

