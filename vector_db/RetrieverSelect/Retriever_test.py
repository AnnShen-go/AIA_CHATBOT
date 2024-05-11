from RetrieverSelect import RetrieverSelect

def main():
    retriever = RetrieverSelect().get_retriever(collection_name="simple_html_Stella_Base_zh_v3_1792d_t13_aia_with_qa")
    test = retriever.invoke("第四期LLM初階班開課時間")

    for x in test['context']:
        print(x.page_content)

if __name__ == '__main__':
    main()