// Bật review MD lên nhé, hoặc xem trực tiếp trên github

# Step - Run
> follow these step or connect author.

## `cd` to recent folder
```Bash
cd 140426
```

## Check python version to make sure everything good
```Bash
where python
# or
where python3
# or py3, with `which` instead of `where`
```

## Run `scripts/bash/first_install.sh` or paste it to terminal
```Bash
python -m pip install pandas scikit-learn streamlit matplotlib seaborn joblib imbalanced-learn kagglehub plotly
```

## Install requirements
```Bash
python -m pip  install -r requirements.txt
```

## Download datasets
```Bash
python -u scripts/python/download_data.py
```
- Dataset tải về nằm ở folder `data`, có 2 set, mình sẽ dùng set chính là bank

## Train, validate and tips
```Bash
python -u src/bank/train_bank.py
```
- Trong này Gemini có comments hết ý nghĩa các hàm và thông số theo những require của a rùi. Có một số option anh có vibe code trong này:
    - Trong folder `repports/bank` có chứa 11 đồ thị thường dùng để đánh giá quá trình train, đánh giá model, em thích dùng cái nào để thuyết trình thì dùng. Nếu thấy không cần thiết hoặc nộp code cho thầy thì cứ xoá ra tương ứng trong code là được, hoặc nhờ anh Tài hoặc anh xoá ( anh xoá thì 500k vnd 1 hàm :D )
    - Trong code có sẵn code đưa lời khuyên, cần thay đổi thông số gì để model tốt hơn
    - Trong code có đoạn đánh giá các kỹ thuật sử dụng trong quá trình train, đánh giá tác dụng và so sánh với các kỹ thuật đồng cấp, em có thể xem để tham khảo thêm hoặc fine-tune theo hướng em thích
    - Kết quả trả về có tham số Accuracy và Brier Score, ở dưới có các tham số precision, precall và f1-score, ... em có thể Gemini, GPT, Claude, ... để hiểu rõ hơn nha. Cái nào cần hiểu thì hãy tìm hiểu, còn ko thì bỏ qua hoặc xoá khỏi code cũng được ha
- Các parameters hay các thông số trong code như: train, test, validate, ... e nhớ để ý để fine-tune model
- Model train xong nằm tại folder `model` dạng .pkl (pickle) hen, e có thể tìm hiểu thêm tại sao lại là định dạng pickle, các định dạng lưu model khác, ưu nhược của từng loại, ... thể thêm kiến thức hen

## Run web demo on local IP
- Chạy demo web local bằng lệnh trong `scripts/bash/run_web.sh` hoặc paste vào terminal:
    ```bash  
    streamlit run src/bank/app_bank.py
    ```
- Web tự mở host tại: `http://localhost:8501/`
- Em có thể test ở đây bằng cách thay đổi thông số bên Hồ sơ Khách hàng, và nhấm `TIẾN HÀNH PHÂN TÍCH CHUYÊN SÂU` để show kết quả nha
- Em có thể dọc file data `data/data_bank.csv` để biết cụ thể data như thế nào
- Website được code trong `src/bank/app_bank.py`, em có thể chỉnh sửa UI/UX trong này, sửa text, để tên mình, tên nhóm, etc. ở đây hen, freestyle đê

## Bảo trì và phát triển
- ginanightbunny@gmail.com
- conquerstellar.polaris
- 0365725913
- Techcombank 16142516112001