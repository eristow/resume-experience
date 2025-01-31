import streamlit as st
from logger import setup_logging
from datetime import datetime, date
from dateutil.relativedelta import relativedelta
from typing import List, Dict, Tuple

logger = setup_logging()


def render_job_input():
    import streamlit as st

    # st.write(st.session_state.enable_dev_features)
    # st.write(st.session_state.using_dev_data)
    # st.write(st.session_state.job_rows)

    if st.session_state.using_dev_data:
        return

    # TODO: add input for full/part time. Assume full-time by default
    def create_job_row(num):
        with st.container(border=True):
            col1, col2, col3 = st.columns(3)
            with col1:
                job_input = st.text_input(f"Job {num}")
            with col2:
                start_date_input = st.date_input(f"Start Date {num}")
            with col3:
                end_date_input = st.date_input(f"End Date {num}")
            job_description = st.text_area(f"Job Description {num}")

            if st.button(f"Delete row {num}", type="primary", icon="❌"):
                return {}

        return {
            "job": job_input,
            "start_date": start_date_input,
            "end_date": end_date_input,
            "description": job_description,
        }

    if st.button("Add new job row"):
        st.session_state.job_rows.append(
            {"job": None, "start_date": None, "end_date": None, "description": ""}
        )

    if st.session_state.enable_dev_features:
        if st.button("Hannah_Resume"):
            st.session_state.using_dev_data = True
            st.session_state.job_rows = [
                {
                    "job": "Breton Bay Country Club, Leonardtown, MD",
                    "start_date": date(2017, 6, 1),
                    "end_date": date(2018, 8, 1),
                    "description": "Lifeguard\n● CPR Certified through American Red Cross\n● In charge of the safety and health of patrons at Breton Bay pool, required teamwork with others as well\nas ability to take charge in an emergency\n● Help with selling of food and drink when not in the guard stand",
                },
                {
                    "job": "Ruddy Duck Brewery & Grill, Dowell, MD",
                    "start_date": date(2019, 5, 1),
                    "end_date": date(2019, 8, 1),
                    "description": "Restaurant Hostess\n● Greeted customers and helped to seat them, managed cash register up front and all to-go orders\n● Helped bus tables, clean around restaurant, and helped serve food when restaurant was busy\n● Required skills in customer service realm, professionalism, and ability to multitask",
                },
                {
                    "job": "Air Combat Effectiveness Consulting Group, LLC, Lexington Park, MD",
                    "start_date": date(2020, 6, 1),
                    "end_date": date(2025, 1, 10),
                    "description": "Administrative Assistant\n● Kept track of inventory for the ACE office in Lexington Park\n● Took messages for and directed phone calls to relevant staff members\n● Processed and directed mail and incoming packages or deliveries\n● Ensured that hard copies and digital copies of important financial documents were properly filed\nand organized\n● Used Adobe Captivate to convert PowerPoint training modules to interactive training modules",
                },
            ]
            return

        if st.button("Garrett Resume"):
            st.session_state.using_dev_data = True
            st.session_state.job_rows = [
                {
                    "job": "Demand Generation – Performance Marketing Manager (Contract) The Walt Disney Company (Disney Publishing – eCom) - Remote",
                    "start_date": date(2021, 5, 1),
                    "end_date": date(2025, 1, 15),
                    "description": "• Google Ads, Microsoft Ads, Google Shopping, Paid Search, Amazon PPC, YouTube Ads & SEO.\n• Top Wins: 48.9K Conversions. Reducing Paid Search CPA from $22.71 to $3.71. A MoM ROAS that fluctuated from $8-$12 starting month 3.\n• Day to Day: Launched & Optimized Performance Marketing Initiatives for The Disney Publishing Division",
                },
                {
                    "job": "SR. Digital Marketing Manager: ECOM, SEO, PPC & Shopping The Home Depot - Remote",
                    "start_date": date(2019, 5, 1),
                    "end_date": date(2025, 1, 15),
                    "description": "Developed and managed comprehensive online marketing campaigns for several divisions, subsidiaries, and company partners. Projects included Google & Bing paid media campaigns, which were optimized based on real- time opportunities.\n• Implemented DSA campaigns, which created a site-wide lift for all over 1M SKUs across three brands.\n• Managed $12M in PPC Ad spend, resulting in $75M in combined online revenues for brand partners including Supply Works, Barnett, Wilmar & Kimberly Clark.\n• Conduct audits & address technical SEO issues maintaining proper indexing, broken links, 404 errors, 301/302 redirects while maintaining proper Canonical Tags, Robotstxt, Meta robot tags & X robots tags.",
                },
                {
                    "job": "Sr. SEO – Digital Marketing Manager Bank of America",
                    "start_date": date(2017, 9, 1),
                    "end_date": date(2019, 5, 1),
                    "description": "Assembled and managed an SEM Strategist team to create competitive campaigns against Square using SEO, PPC, Facebook and YouTube Ads. Served as SEO subject matter expert to product IT, Dev Ops, and Copywriters, providing SEO recommendations.\n• Created, managed and executed SEO, SEM, PPC & display campaigns for TOF, MOF & BOF programs.\n• Performed technical SEO & PPC audits, conversion rate optimization audits, competitor website & SEM audits.\n• Conducted content gap analysis reports while ensuring lower CPCs, higher CTR and Increasing ROAS Targets.",
                },
                {
                    "job": "Senior Digital Marketer SEO, PPC Strategist, Digital Analytics Thomson Reuters",
                    "start_date": date(2015, 7, 1),
                    "end_date": date(2017, 9, 1),
                    "description": "Acted as the Digital Strategist for FindLaw, a part of Thomson Reuters, managing PPC and paid search\ncampaigns for over 63 law firms. Created and executed a holistic search strategy (SEO & PPC) supporting\nthe clients’ business objectives. Created lead generation campaigns through consultation form fills and\nclick to call campaigns.\n• Reduced T-CPA by 30% for CPC’s like “Personal Injury” by creating SKAGS and GEO-Modifiers.\n• Achieved a PI Case Acquisition cost of $2K.\n• Generated over 700 new signed personal injury cases within the first five months of the campaign.\n• Scaled the monthly budget from $60k to $350K.",
                },
                {
                    "job": "Digital Marketing Manager SEO By Nerds",
                    "start_date": date(2010, 2, 1),
                    "end_date": date(2015, 7, 1),
                    "description": "Directed online marketing activities for over 300 websites and over 5K search terms. Industries included\nLaw\nFirms, Dental Practices, Medical Offices, Franchises, Multi-Location Businesses and Fortune 100-500\ncompanies.\n• Managed digital campaign assets, landing page development and content, and tracking that support\ndirect response marketing strategy.\n• Collaborate with marketing team partners to develop new paid search campaign content and messaging\nto achieve business objectives.\n• Performed continuous A/B and multivariate testing of campaign messaging and landing pages.\n• Evaluated and drove qualitative KPIs through continuous improvement and optimization.",
                },
            ]
            return

    for i, row in enumerate(st.session_state.job_rows):
        new_row = create_job_row(i + 1)

        if len(new_row.keys()) == 0:
            del st.session_state.job_rows[i]
        else:
            st.session_state.job_rows[i] = new_row
