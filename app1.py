import streamlit as st
import pandas as pd
import requests
import json
import joblib

# ==========================================
# إعدادات صفحة الويب
# ==========================================
st.set_page_config(page_title="Zola Football Predictor", page_icon="⚽", layout="wide")

st.title("⚽ منصة توقعات كرة القدم واقتناص الفرص")
st.markdown("هذه المنصة تستخدم نموذج **XGBoost** مدرب على بيانات تاريخية منذ عام 1993 لتوقع نتائج مباريات الدوري الإنجليزي الممتاز ومقارنتها مع احتمالات مكاتب المراهنات لاستخراج الفرص الذهبية (Value Bets).")

# ==========================================
# تحميل النموذج والملفات (مع خاصية الكاش لتسريع الموقع)
# ==========================================
@st.cache_resource
def load_assets():
    try:
        model = joblib.load('football_xgboost_model.pkl')
        elo_dict = joblib.load('teams_elo_ratings.pkl')
        with open('teams_master_map.json', 'r', encoding='utf-8') as f:
            team_map = json.load(f)
        return model, elo_dict, team_map
    except Exception as e:
        st.error(f"حدث خطأ أثناء تحميل ملفات النموذج: {e}")
        return None, None, None

model, elo_dict, team_map = load_assets()

def get_standard_name(name):
    return team_map.get(name, name)

def get_elo(team_name):
    return elo_dict.get(team_name, 1500)

# ==========================================
# دالة جلب البيانات من الـ API
# ==========================================
def fetch_upcoming_matches_and_odds(api_key):
    url = f"https://api.the-odds-api.com/v4/sports/soccer_epl/odds/?apiKey={api_key}&regions=uk&markets=h2h"
    response = requests.get(url)
    
    if response.status_code != 200:
        st.error("❌ فشل الاتصال بـ Odds API. تأكد من صحة المفتاح أو رصيد الباقة.")
        return []

    matches = []
    for event in response.json():
        if not event.get('bookmakers'):
            continue
            
        bookmaker = event['bookmakers'][0] 
        market = bookmaker['markets'][0]
        
        odds_data = {outcome['name']: outcome['price'] for outcome in market['outcomes']}
            
        matches.append({
            'Home': event['home_team'],
            'Away': event['away_team'],
            'Home_Odds': odds_data.get(event['home_team'], 0),
            'Draw_Odds': odds_data.get('Draw', 0),
            'Away_Odds': odds_data.get(event['away_team'], 0)
        })
    return matches

# ==========================================
# واجهة التفاعل والتوقعات
# ==========================================
# جلب المفتاح بأمان من إعدادات Streamlit
try:
    ODDS_API_KEY = st.secrets["ODDS_API_KEY"]
except KeyError:
    st.warning("⚠️ مفتاح Odds API غير موجود. يرجى إضافته في إعدادات Streamlit Secrets.")
    ODDS_API_KEY = None

if st.button("🔍 تحليل مباريات الجولة القادمة", type="primary"):
    if not ODDS_API_KEY:
        st.error("لا يمكن الاستمرار بدون مفتاح الـ API.")
    elif model and elo_dict and team_map:
        with st.spinner('جاري جلب البيانات وتحليل الاحتمالات...'):
            upcoming_matches = fetch_upcoming_matches_and_odds(ODDS_API_KEY)
            
            if upcoming_matches:
                results_list = []
                
                for match in upcoming_matches:
                    home_std = get_standard_name(match['Home'])
                    away_std = get_standard_name(match['Away'])
                    
                    home_elo = get_elo(home_std)
                    away_elo = get_elo(away_std)
                    elo_diff = home_elo - away_elo
                    
                    match_features = pd.DataFrame([{
                        'Home_Elo': home_elo,
                        'Away_Elo': away_elo,
                        'Elo_Difference': elo_diff
                    }])
                    
                    model_probs = model.predict_proba(match_features)[0]
                    prob_home_win = model_probs[2]
                    
                    implied_home_prob = 1 / match['Home_Odds'] if match['Home_Odds'] > 0 else 0
                    value_edge = prob_home_win - implied_home_prob
                    is_value_bet = value_edge > 0.05 
                    
                    results_list.append({
                        'المباراة': f"{home_std} (أرضه) ضد {away_std}",
                        'توقع الذكاء الاصطناعي': f"{prob_home_win*100:.1f}%",
                        'توقع السوق (الاحتمالات)': f"{implied_home_prob*100:.1f}%",
                        'نسبة التفوق': f"{value_edge*100:.1f}%",
                        'فرصة ذهبية؟': '✅ نعم' if is_value_bet else '❌ لا'
                    })

                # عرض النتائج في جدول تفاعلي
                df_results = pd.DataFrame(results_list)
                st.success("تم الانتهاء من التحليل!")
                st.dataframe(df_results, use_container_width=True)
            else:
                st.info("لا توجد مباريات قادمة أو احتمالات متاحة حالياً.")
