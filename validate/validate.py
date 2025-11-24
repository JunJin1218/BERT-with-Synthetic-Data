# 여기서는 ./models/의 모델들을 불러와서 평가
# ./models/에는
# ./models/{task}/{ratio}/ 폴더들이 있음 {ratio}는 sythetic data : real data의 비율을 의미함
# 1. {unique_model_name_based_on_time}.pt 등의 파라미터랑
# 2. {unique_model_name_based_on_time}.json 등으로 모델을 훈련했을 때의 하이퍼 파라미터 및 세팅 로그
# ./setting.yaml에서 config 불러와서 (hydra - DictConfig인가 그걸로 불러오기)
# 거기서 각 폴더별로 best validated model을 뽑아오고 저장하면 될듯?