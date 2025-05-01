import pandas as pd

# 假設你的資料集已經載入到 df 中
df = pd.read_csv(r"D:\datascience\Sec_homework\dataset\test.csv")

# 定義中文星座對應的數字
star_sign_mapping = {
    "牡羊座": 1,   # Aries
    "金牛座": 2,   # Taurus
    "雙子座": 3,   # Gemini
    "巨蟹座": 4,   # Cancer
    "獅子座": 5,   # Leo
    "處女座": 6,   # Virgo
    "天秤座": 7,   # Libra
    "天蠍座": 8,   # Scorpio
    "射手座": 9,   # Sagittarius
    "魔羯座": 10,  # Capricorn
    "水瓶座": 11,  # Aquarius
    "雙魚座": 12   # Pisces
}

# 直接將映射結果覆寫回 star_sign 欄位
df["star_sign"] = df["star_sign"].map(star_sign_mapping)

# 如有需要，可儲存或進一步處理資料
df.to_csv("mapped_dataset.csv", index=False)
