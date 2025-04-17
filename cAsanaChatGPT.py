import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
from openai import OpenAI
from dotenv import load_dotenv
import os
from datetime import datetime
from io import BytesIO
import gspread
from google.oauth2 import service_account

st.set_page_config(page_title="تقرير مشروع ", layout="centered")

# تحميل المفاتيح
load_dotenv()
ASANA_TOKEN = os.getenv("ASANA_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
headers = {"Authorization": f"Bearer {ASANA_TOKEN}"}

# --------- تصميم --------- #
st.markdown("""
    <style>
        body {background-color: #f9fafb; direction: rtl; padding: 10px; font-size: 15px;}
        .report-title {color: #1d4ed8; font-size: 36px; font-weight: bold; margin-bottom: 20px;}
        .section-header {color: #1f2937; font-size: 24px; font-weight: bold; margin-top: 40px; margin-bottom: 15px;}
        .gpt-summary-box {background-color: #f3f4f6; padding: 20px; border-radius: 10px; font-weight: 500;}
        .dataframe th {background-color: #1d4ed8; text-align: right;}
        .stButton button {background-color: #861d1d; color: white; font-weight: bold; border-radius: 8px; padding: 10px 20px;}
        .stButton button:hover {background-color: #861d1d;}
        .print-button {margin-top: 40px; text-align: center;}
    </style>
""", unsafe_allow_html=True)


# --------- الوظائف --------- #
def get_project_name(project_id):
    url = f"https://app.asana.com/api/1.0/projects/{project_id}"
    response = requests.get(url, headers=headers)
    return response.json()['data']['name']

def get_asana_tasks(project_id):
    url = f"https://app.asana.com/api/1.0/projects/{project_id}/tasks?opt_fields=name,assignee.name,created_by.name,completed,due_on,gid"
    response = requests.get(url, headers=headers)
    return response.json()['data']

def process_tasks_to_df(tasks, project_id):
    data = []
    for task in tasks:
        data.append({
            "المهمة": task['name'],
            "المسند إليه": task["assignee"]["name"] if task.get("assignee") else "غير مسند",
            "أنشأها": task["created_by"]["name"] if task.get("created_by") else "غير معروف",
            "الحالة": "✅" if task.get("completed") else "❌",
            "تاريخ التسليم": task.get("due_on")
        })
    df = pd.DataFrame(data)
    df["تاريخ التسليم"] = pd.to_datetime(df["تاريخ التسليم"], errors='coerce')
    return df

def generate_user_stats(df):
    df_copy = df.copy()
    df_copy["متأخرة"] = (df_copy["الحالة"] == "❌") & (df_copy["تاريخ التسليم"] < pd.Timestamp.today())
    grouped = df_copy.groupby("المسند إليه").agg({
        "المهمة": "count",
        "الحالة": lambda x: (x == "✅").sum(),
        "متأخرة": "sum"
    }).reset_index()
    grouped.columns = ["المسند إليه", "عدد المهام", "المهام المكتملة", "المهام المتأخرة"]
    return grouped

def generate_created_by_stats(df):
    return df.groupby("أنشأها").agg({"المهمة": "count"}).reset_index().rename(columns={"المهمة": "عدد المهام المنشأة"})

def generate_summary_with_gpt(df, user_stats):
    client = OpenAI(api_key=OPENAI_API_KEY)
    prompt = f"""
أنت مساعد إداري ذكي. هذا تقرير بيانات مهام مشروع في Asana:

أول 10 مهام:
{df.head(10).to_string(index=False)}

تحليل الموظفين:
{user_stats.to_string(index=False)}

- عدد المهام: {len(df)}
- مكتملة: {(df['الحالة'] == '✅').sum()}
- غير مكتملة: {(df['الحالة'] == '❌').sum()}
- متأخرة: {(df['الحالة'] == '❌') & (df['تاريخ التسليم'] < pd.Timestamp.today()).sum()}

ابدأ بملخص عام عن المشروع.
ثم افصل بين:
- نقاط القوة
- نقاط الضعف
- التوصيات
واكتب عنوان واضح قبل كل قسم.
"""
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4
    )
    return response.choices[0].message.content

def generate_chart(df):
    counts = {
        "Completed": (df["الحالة"] == "✅").sum(),
        "Not completed ": (df["الحالة"] == "❌").sum(),
        "Delayed": ((df["الحالة"] == "❌") & (df["تاريخ التسليم"] < pd.Timestamp.today())).sum()
    }
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.bar(counts.keys(), counts.values(), color=['green', 'orange', 'red'])
    ax.set_title("Tasks status ")
    ax.set_ylabel("Number of tasks ")
    st.pyplot(fig)

def export_excel_report(df, user_stats_df, created_by_df, summary, project_name):
    overall_summary = pd.DataFrame({
        "اسم المشروع": [project_name],
        "عدد المهام": [len(df)],
        "مكتملة": [(df["الحالة"] == "✅").sum()],
        "غير مكتملة": [(df["الحالة"] == "❌").sum()],
        "متأخرة": [((df["الحالة"] == "❌") & (df["تاريخ التسليم"] < pd.Timestamp.today())).sum()]
    })

    summary_df = pd.DataFrame({"تحليل GPT": [summary]})
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name="مهام المشروع")
        overall_summary.to_excel(writer, index=False, sheet_name="إحصائيات عامة")
        user_stats_df.to_excel(writer, index=False, sheet_name="تحليل الموظفين")
        created_by_df.to_excel(writer, index=False, sheet_name="منشئي المهام")
        summary_df.to_excel(writer, index=False, sheet_name="تحليل GPT")
    output.seek(0)
    return output

def upload_to_google_sheets(sheet_id, df_tasks, df_summary, df_users, df_created_by, gpt_summary):
    scopes = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
    credentials_dict = st.secrets["google_service_account"]
    credentials = service_account.Credentials.from_service_account_info(credentials_dict, scopes=scopes)
    client = gspread.authorize(credentials)
    sheet = client.open_by_key(sheet_id)

    df_tasks = df_tasks.fillna("").astype(str)
    df_summary = df_summary.fillna("").astype(str)
    df_users = df_users.fillna("").astype(str)
    df_created_by = df_created_by.fillna("").astype(str)

    def update_sheet(name, data):
        try:
            ws = sheet.worksheet(name)
            ws.clear()
        except:
            ws = sheet.add_worksheet(title=name, rows="100", cols="20")
        ws.update([data.columns.values.tolist()] + data.values.tolist())

    update_sheet("مهام المشروع", df_tasks)
    update_sheet("إحصائيات عامة", df_summary)
    update_sheet("تحليل الموظفين", df_users)
    update_sheet("منشئي المهام", df_created_by)

    try:
        ws = sheet.worksheet("تحليل GPT")
        ws.clear()
    except:
        ws = sheet.add_worksheet(title="تحليل GPT", rows="100", cols="1")
    ws.update("A1", [[gpt_summary]])

# --------- واجهة المستخدم --------- #
st.markdown("<div class='report-title'>📋 مولد تقرير Asana</div>", unsafe_allow_html=True)
project_id_input = st.text_input("🔢 معرّف المشروع (Project ID)", "")

if st.button("..توليد التقرير") and project_id_input:
    with st.spinner("📡 جاري تحميل البيانات..."):
        try:
            project_name = get_project_name(project_id_input)
            tasks = get_asana_tasks(project_id_input)
            df = process_tasks_to_df(tasks, project_id_input)
            user_stats_df = generate_user_stats(df)
            created_by_stats_df = generate_created_by_stats(df)
            summary = generate_summary_with_gpt(df, user_stats_df)

            st.markdown(f"<div class='section-header'>📌 تقرير المشروع: {project_name}</div>", unsafe_allow_html=True)
            st.markdown(f"<p class='text-gray-600'>🕒 التاريخ: {datetime.now().strftime('%Y-%m-%d')}</p>", unsafe_allow_html=True)

            st.markdown("<div class='section-header'>🧪 التحليل الذكي</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='gpt-summary-box'>{summary}</div>", unsafe_allow_html=True)

            st.markdown("<div class='section-header'>📈 الرسم البياني</div>", unsafe_allow_html=True)
            generate_chart(df)

            
            st.markdown("<div class='section-header'>📊 المهام</div>", unsafe_allow_html=True)
            st.dataframe(df, use_container_width=True)




            st.markdown("<div class='section-header'>👥 الموظفون</div>", unsafe_allow_html=True)
            st.dataframe(user_stats_df, use_container_width=True)

            st.markdown("<div class='section-header'>📝 عدد المهام التي أنشأها كل موظف</div>", unsafe_allow_html=True)
            st.dataframe(created_by_stats_df, use_container_width=True)

            overall_summary = pd.DataFrame({
                "اسم المشروع": [project_name],
                "عدد المهام": [len(df)],
                "مكتملة": [(df["الحالة"] == "✅").sum()],
                "غير مكتملة": [(df["الحالة"] == "❌").sum()],
                "متأخرة": [((df["الحالة"] == "❌") & (df["تاريخ التسليم"] < pd.Timestamp.today())).sum()]
            })

            upload_to_google_sheets(
                sheet_id=st.secrets["GOOGLE_SHEET_ID"],
                df_tasks=df,
                df_summary=overall_summary,
                df_users=user_stats_df,
                df_created_by=created_by_stats_df,
                gpt_summary=summary
            )

            st.success("✅ تم تحديث Google Sheets بنجاح!")

            st.markdown("<div class='section-header'>📥 تحميل ملف Excel</div>", unsafe_allow_html=True)
            excel_file = export_excel_report(df, user_stats_df, created_by_stats_df, summary, project_name)
            st.download_button(
                label="⬇️ تحميل تقرير Excel",
                data=excel_file,
                file_name=f"asana_report_{datetime.now().strftime('%Y%m%d')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

        except Exception as e:
            st.error(f"❌ حدث خطأ: {e}")
