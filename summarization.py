from duckduckgo_search import DDGS
from fastapi.responses import JSONResponse
import pandas as pd
import time
import re
import random

# 데이터 프레임 전처리
df = pd.read_csv('/home/kkm/work/flyai/K-CAT/kkm/preprocessing/datasets/PJZ.csv', encoding="latin1")
df = df.drop(df.columns[6:], axis=1)
df_0 = df[df['label'] == 0] # 일상 대화
df_0.reset_index(drop=True, inplace=True)

print(max(df_0['message_idx'].value_counts())) # 149이므로, 나눌 덩어리(청크) 개수를 넉넉하게 6으로
num_of_chunks = 5
num_of_chats = 4
max_message_idx = max(df_0['message_idx']) # 채팅방 단위로 구별해야 되니까, message_idx의 최대값을 구한 뒤 모든 채팅방 인덱스에 접근

# 채팅방 인덱스 조사: 채팅방 단위로 구별해야 되니까, message_idx의 최대/최소값을 구한 뒤 모든 채팅방 인덱스에 접근
min_message_idx = min(df_0['message_idx'])
print(min_message_idx)
max_message_idx = max(df_0['message_idx'])
print(max_message_idx)

# 초기화
total_df_0_idx = 0
ddgs_session = DDGS()
json_exclude_chars = r'[\{\[\(\)\]\}\,\":\\]' # JSON parser에게 혼동될 문자들 이스케이프 처리
num_of_translated_sentences = 0

# 전처리 시작
for i in range(min_message_idx, max_message_idx+1):
    try:
        summarized_chat = ""
        total_chat_length = 0
        df_0_one_chat = pd.DataFrame()
        print('********************************************************************')
        print(f'채팅방 인덱스: {i}')
        df_0_one_chat = df_0[df_0['message_idx'] == i].copy()
        df_0_one_chat.reset_index(drop=True, inplace=True)
        
        # 채팅방 4개에 해당하는 문장들 다 넣었을 때마다 gpt 대화세션 새로 파기
        if i % num_of_chats == 0:
            ddgs_session = DDGS()

        # 현재 채팅방의 문장 개수 구하고, 한 번에 넣을 문장 개수를 의미하는 offset 만큼씩 넣고 번역한 것을 받아오기
        total_chat_length = df_0_one_chat.shape[0]
        offset = total_chat_length // num_of_chunks # 최대토큰수가 넘어갈 수도 있으니까 총 길이의 1/num_of_chunks씩 넣기
        print(f'채팅방 {i}의 문장 수: {total_chat_length}, offset: {offset}')
        
        # 채팅방 문장 수가 덩어리 수로 나누어 떨어지지 않으면 한번 더 반복해서 나머지 문장들도 넣어주기 위한 max_chunk_iter
        max_chunk_iter = 1
        if total_chat_length > num_of_chunks * 2: # 문장 수 10개 초과할 때에만 덩어리로 나눔
            if total_chat_length % num_of_chunks == 0: # num_of_chunks=5
                max_chunk_iter = num_of_chunks
            else: 
                max_chunk_iter = num_of_chunks + 1
        print(f'max_chunk_iter: {max_chunk_iter}')
        
        for j in range(max_chunk_iter):
            chat_chunk = ""
            query_prompt = ""
            if j == 0:
                query_prompt = "내가 입력하는 새로운 대화 상황들을 한국어로 번역한 요약본을 생성해줘.\
                                대화 상황들을 덩어리 단위로 끊어서 입력할건데, 입력한 모든 덩어리들의 종합적인 문맥을 고려해야 해.\
                                각 덩어리들은 두 개의 %사이에 시간이 명시되어 있고, ~문자에 이어서 대화한 사람의 이름이 써있어.\
                                입력할 덩어리가 끝났으면 따로 말해줄게. 그 전까지는 덩어리 문장을 받으면 알겠습니다.로 대답만 해줘\n"
            if j == max_chunk_iter - 1:
                query_prompt = "덩어리를 다 입력했어. 그럼 이전에 입력했던 문장 덩어리들을 한꺼번에 고려해서 한국어로 번역한 요약본을 생성해줘.\
                                대답하지 말고 바로 번역해줘. 인사는 하지 말고.\n"
            else:
                query_prompt = "덩어리 계속 입력할게. 각 덩어리들은 두 개의 %사이에 시간이 명시되어 있고, 이어서 대화한 사람의 이름이 써있는거 잊지마.\
                                알겠으면 알겠습니다.라고만 대답해줘.\n"
                                
            
            # 청크덩어리 하나에 있는 번역할 문장들 chat_chunk에 넣기
            if j < max_chunk_iter - 1:
                max_sentence_iter = offset
            else: # num_of_chunks개의 청크 중 마지막 청크는 남은거 모두 넣기
                max_sentence_iter = total_chat_length - j*offset
            print(f'max_sentence_iter: {max_sentence_iter}')
            
            for k in range(max_sentence_iter):
                print(f'current idx: {j*offset + k}, df_0_one_chat shape:{df_0_one_chat.shape}')
                chat_chunk += '%' + df_0_one_chat.loc[j*offset + k, 'time'] + '%~' + df_0_one_chat.loc[j*offset + k, 'author'] + ":" \
                    + df_0_one_chat.loc[j*offset + k, 'text'] + '\n'
            
            # 정규표현식을 이용해 이스케이프 처리
            chat_chunk = re.sub(json_exclude_chars, lambda match: '\\' + match.group(0), chat_chunk)
            
            # 번역 및 요약 요청 전송
            if j < max_chunk_iter - 1:
                reply = ddgs_session.chat(query_prompt + chat_chunk, model='gpt-4o-mini',timeout=500)
                print(reply)
                time.sleep(random.randint(4, 6))
            else:
                summarized_chat = ddgs_session.chat(query_prompt + chat_chunk, model='gpt-4o-mini',timeout=500)
                time.sleep(random.randint(4, 6))
            
        # 이스케이프 없애기(혹시나)
        summarized_chat= re.sub(r'\\([{}\[\]()\",:])', r'\1', summarized_chat)
        
        print(summarized_chat)
            
    except Exception as e:
        print(e)
        print(f'에러발생. message_idx {i}번째 채팅 번역중')
        df_0.to_csv(f'./datasets/PJZ_summarized_save_{i}.csv', encoding='utf-8-sig')
        break
    
    df_0.loc[total_df_0_idx, 'translated chat summary'] = summarized_chat
    total_df_0_idx += total_chat_length # df_0에 저장할 위치 찾기위한 인덱스 total_df_0_idx
            
    
df_1 = df[df['label'] == 1]
df_out = pd.concat([df_1, df_0], axis=0)
df_out.to_csv(f'PJZ_summaryEDA.csv', encoding='utf-8-sig')