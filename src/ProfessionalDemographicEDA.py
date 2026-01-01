class ProfessionalDemographicEDA:
    """
    ROFESSIONAL DEMOGRAPHIC EDA — CLIENT-READY & HIGH-IMPACT
    - 3 Key Visualizations: Education, e-KTP Coverage, & Population Trends
    - 3 Strategic Tables: Priority Villages, Critical Districts, & Trend Summaries
    - Focuses on operational recommendations and strategic insights.
    """

    def __init__(self, input_dir: str = "ultimate_clean_output", output_dir: str = "final_professional_eda"):
        self.input_dir = input_dir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Set Professional Plotting Styles
        try:
            plt.style.use('seaborn-v0_8-whitegrid')
        except:
            plt.style.use('ggplot')
            
        sns.set_palette("husl")
        plt.rcParams.update({
            'font.family': 'DejaVu Sans',
            'savefig.dpi': 300,
            'figure.dpi': 150,
            'axes.titlesize': 18,
            'axes.labelsize': 14,
            'xtick.labelsize': 12,
            'ytick.labelsize': 12
        })

        print(f"📁 Output directory initialized: {output_dir}")

    def generate_professional_eda(self) -> bool:
        """Main execution flow for generating professional EDA assets."""
        print("🎯 Starting Professional EDA — FOCUSING ON STRATEGIC INSIGHTS...")

        # Data Loading
        df_kel = self._safe_load("profil_kelurahan_clean.csv")
        df_kec = self._safe_load("profil_kecamatan_clean.csv")
        df_tren = self._safe_load("tren_kota_clean.csv")

        if df_kel is None or df_kec is None or df_tren is None:
            print("❌ Incomplete data — Process halted.")
            return False

        if len(df_kel) == 0 or len(df_kec) == 0 or len(df_tren) == 0:
            print("❌ Empty datasets detected — Process halted.")
            return False

        print(f"✅ Data Ready: Village Records ({len(df_kel):,}), District Records ({len(df_kec):,}), Trends ({len(df_tren)})")

        # === 1. STRATEGIC VISUALIZATIONS ===
        print("\n🖼️  Generating 3 Strategic Visualizations...")
        self._plot_top10_village_higher_edu(df_kel)
        self._plot_e_ktp_coverage_comparison(df_kec)
        self._plot_population_growth_trend(df_tren)

        # === 2. STRATEGIC TABLES ===
        print("\n📊 Generating 3 Strategic Tables...")
        table1 = self._generate_village_priority_table(df_kel)
        table2 = self._generate_district_critical_table(df_kec)
        table3 = self._generate_city_trend_summary_table(df_tren)

        # Export Tables
        table1.to_csv(os.path.join(self.output_dir, "strategic_village_priority.csv"), index=False)
        table2.to_csv(os.path.join(self.output_dir, "strategic_district_critical.csv"), index=False)
        table3.to_csv(os.path.join(self.output_dir, "strategic_city_trend_summary.csv"), index=False)

        print(f"✅ Strategic tables saved to {self.output_dir}")

        # === 3. EXECUTIVE SUMMARY ===
        self._export_executive_summary(table1, table2, table3)

        print(f"\n🎉 PROFESSIONAL EDA COMPLETE — CLIENT PRESENTATION READY!")
        return True

    def _safe_load(self, filename: str) -> Optional[pd.DataFrame]:
        """Utility for safe data loading."""
        filepath = os.path.join(self.input_dir, filename)
        if not os.path.exists(filepath):
            print(f"❌ File {filename} not found in {self.input_dir}")
            return None
        try:
            return pd.read_csv(filepath)
        except Exception as e:
            print(f"❌ Failed to load {filename}: {e}")
            return None

    def _plot_top10_village_higher_edu(self, df_kel: pd.DataFrame):
        """Plots Top 10 Villages by Higher Education percentage vs National Benchmark."""
        higher_edu_cats = ['DIPLOMA IV/STRATA I', 'STRATA 2', 'STRATA 3']
        df_edu = df_kel[(df_kel['tipe_data'] == 'jenis_pendidikan') & (df_kel['kategori'].isin(higher_edu_cats))]

        edu_counts = df_edu.groupby('kemendagri_nama_desa_kelurahan')['jumlah'].mean().reset_index()
        pop_counts = df_kel.groupby('kemendagri_nama_desa_kelurahan')['jumlah_penduduk'].mean().reset_index()
        combined = edu_counts.merge(pop_counts, on='kemendagri_nama_desa_kelurahan')
        combined['edu_pct'] = (combined['jumlah'] / combined['jumlah_penduduk'] * 100).round(2)

        top_10 = combined.nlargest(10, 'edu_pct').copy()
        nat_avg = 12.5 # National Benchmark

        top_10['tier'] = np.where(top_10['edu_pct'] > 20, '🟢 Excellent',
                         np.where(top_10['edu_pct'] >= nat_avg, '🟡 Good', '🔴 Below Avg'))
        top_10['rank'] = range(1, 11)

        plt.figure(figsize=(16, 9))
        colors = ['#2ca02c' if t == '🟢 Excellent' else '#dbdb8d' if t == '🟡 Good' else '#d62728' for t in top_10['tier']]
        bars = plt.barh(range(len(top_10)), top_10['edu_pct'], color=colors, edgecolor='black', alpha=0.8)
        
        plt.yticks(range(len(top_10)), [f"{r}. {n}" for r, n in zip(top_10['rank'], top_10['kemendagri_nama_desa_kelurahan'])])
        plt.title('Top 10 Villages: Higher Education Percentage\n(vs National Average Benchmark)', fontweight='bold')
        plt.xlabel('Higher Education Percentage (%)')
        plt.axvline(x=nat_avg, color='grey', linestyle='--', label=f'National Avg ({nat_avg}%)')
        plt.legend(loc='lower right')

        for i, (bar, val, tier) in enumerate(zip(bars, top_10['edu_pct'], top_10['tier'])):
            plt.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2, f'{val:.1f}%\n{tier}', va='center', fontweight='bold')

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "top10_village_education.png"))
        plt.close()

    def _plot_e_ktp_coverage_comparison(self, df_kec: pd.DataFrame):
        """Visualizes e-KTP coverage gaps across districts."""
        if 'coverage_e_ktp' not in df_kec.columns: return

        kec_cov = df_kec.groupby('kemendagri_nama_kecamatan')['coverage_e_ktp'].mean().reset_index()
        kec_cov = kec_cov.sort_values('coverage_e_ktp')
        
        plot_data = pd.concat([kec_cov.head(5), kec_cov.tail(5)], ignore_index=True)
        plot_data['gap'] = 90 - plot_data['coverage_e_ktp']

        plt.figure(figsize=(16, 9))
        norm = plt.Normalize(plot_data['gap'].min(), plot_data['gap'].max())
        colors = plt.cm.RdYlGn_r(norm(plot_data['gap']))
        
        bars = plt.barh(range(len(plot_data)), plot_data['coverage_e_ktp'], color=colors, edgecolor='black')
        plt.yticks(range(len(plot_data)), plot_data['kemendagri_nama_kecamatan'])
        plt.title('e-KTP Coverage per District: Gap Analysis vs 90% Target\n(Bottom 5 & Top 5 Districts)', fontweight='bold')
        plt.axvline(x=90, color='orange', linestyle='--', label='Operational Target (90%)')
        plt.legend()

        for i, (bar, val, gap) in enumerate(zip(bars, plot_data['coverage_e_ktp'], plot_data['gap'])):
            plt.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2, f'{val:.1f}% (Gap: {gap:.1f})', va='center', fontweight='bold')

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "e_ktp_coverage_analysis.png"))
        plt.close()

    def _plot_population_growth_trend(self, df_tren: pd.DataFrame):
        """Plots population trends with 2025 Linear Regression prediction."""
        df_tren = df_tren.sort_values('tahun').reset_index(drop=True)
        
        X = df_tren['tahun'].values.reshape(-1, 1)
        y = df_tren['jumlah_penduduk'].values
        model = LinearRegression().fit(X, y)
        
        pred_2025 = model.predict([[2025]])[0]
        std_resid = np.std(y - model.predict(X))

        plt.figure(figsize=(16, 9))
        plt.plot(df_tren['tahun'], df_tren['jumlah_penduduk'], marker='o', lw=3, label='Actual Data')
        plt.plot([df_tren['tahun'].iloc[-1], 2025], [df_tren['jumlah_penduduk'].iloc[-1], pred_2025], 
                 '--s', lw=3, color='#ff7f0e', label='2025 Forecast')
        
        plt.fill_between([df_tren['tahun'].iloc[-1], 2025], [pred_2025 - std_resid]*2, [pred_2025 + std_resid]*2, 
                         color='#ff7f0e', alpha=0.15, label='Confidence Interval (±1 SD)')

        plt.title('Bandung City Population Growth Trend\n(Actual Records + 2025 Forecast)', fontweight='bold')
        plt.ylabel('Population Count')
        plt.xlabel('Year')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "population_trend_forecast.png"))
        plt.close()

    def _generate_village_priority_table(self, df_kel: pd.DataFrame) -> pd.DataFrame:
        """Table: High-growth villages requiring educational intervention."""
        # Calculate Growth Rate
        growth = df_kel.groupby(['kemendagri_nama_desa_kelurahan', 'tahun'])['jumlah_penduduk'].mean().unstack()
        growth_rate = (((growth.iloc[:, -1] - growth.iloc[:, 0]) / growth.iloc[:, 0]) * 100).round(2) if growth.shape[1] >= 2 else 0

        # Calculate Education Index
        higher_edu = ['DIPLOMA IV/STRATA I', 'STRATA 2', 'STRATA 3']
        edu_df = df_kel[(df_kel['tipe_data'] == 'jenis_pendidikan') & (df_kel['kategori'].isin(higher_edu))]
        edu_index = (edu_df.groupby('kemendagri_nama_desa_kelurahan')['jumlah'].sum() / 
                     df_kel.groupby('kemendagri_nama_desa_kelurahan')['jumlah_penduduk'].mean() * 100).round(2)

        result = pd.DataFrame({
            'Village': growth.index,
            'Growth_Rate_Pct': growth_rate,
            'Higher_Edu_Index_Pct': edu_index
        }).dropna()

        result['Risk_Category'] = result.apply(
            lambda x: '🟠 Medium (Priority)' if x['Growth_Rate_Pct'] > 5 and x['Higher_Edu_Index_Pct'] < 15 
            else ('🟢 Low' if x['Growth_Rate_Pct'] > 5 else '🔴 High'), axis=1
        )

        return result.nlargest(10, 'Growth_Rate_Pct').reset_index(drop=True)

    def _generate_district_critical_table(self, df_kec: pd.DataFrame) -> pd.DataFrame:
        """Table: Districts with highest administrative gaps (e-KTP)."""
        pop = df_kec.groupby('kemendagri_nama_kecamatan')['jumlah_penduduk'].mean()
        ktp = df_kec.groupby('kemendagri_nama_kecamatan')['coverage_e_ktp'].mean() if 'coverage_e_ktp' in df_kec.columns else 0
        
        result = pd.DataFrame({
            'District': pop.index,
            'Avg_Population': pop.astype(int),
            'eKTP_Coverage_Pct': ktp.round(2)
        })
        
        result['Est_Unregistered_Pop'] = ((100 - result['eKTP_Coverage_Pct']) * result['Avg_Population'] / 100).astype(int)
        return result.nlargest(5, 'Est_Unregistered_Pop').reset_index(drop=True)

    def _generate_city_trend_summary_table(self, df_tren: pd.DataFrame) -> pd.DataFrame:
        """Table: Clean annual population growth summary."""
        df = df_tren.sort_values('tahun').copy()
        df['Growth_Abs'] = df['jumlah_penduduk'].diff().fillna(0).astype(int)
        df['Growth_Pct'] = (df['jumlah_penduduk'].pct_change() * 100).round(2).fillna(0)
        return df[['tahun', 'jumlah_penduduk', 'Growth_Abs', 'Growth_Pct']].rename(columns={'tahun': 'Year', 'jumlah_penduduk': 'Population'})

    def _export_executive_summary(self, t1: pd.DataFrame, t2: pd.DataFrame, t3: pd.DataFrame):
        """Exports a strategic executive summary for stakeholders."""
        highest_v = t1.iloc[0]['Village']
        lowest_k = t2.iloc[0]['District']
        
        with open(os.path.join(self.output_dir, "EXECUTIVE_SUMMARY.txt"), 'w', encoding='utf-8') as f:
            f.write("📊 EXECUTIVE SUMMARY - BANDUNG DEMOGRAPHIC STRATEGIC ANALYSIS\n")
            f.write("="*65 + f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("💡 KEY INSIGHTS:\n")
            f.write(f" • Village '{highest_v}' shows peak growth ({t1.iloc[0]['Growth_Rate_Pct']}%). Educational infrastructure expansion is vital.\n")
            f.write(f" • District '{lowest_k}' has an estimated {t2.iloc[0]['Est_Unregistered_Pop']:,} residents without e-KTPs. This is an operational bottleneck.\n")
            
            f.write("\n🎯 STRATEGIC RECOMMENDATIONS:\n")
            f.write(f" • Launch 'Localized Education Grants' in {highest_v} to bridge the human capital gap.\n")
            f.write(f" • Deploy Mobile ID Units in {lowest_k} to hit the 90% coverage target within 6 months.\n")
            f.write(f" • Revise the 2025 Infrastructure Masterplan to accommodate forecasted population density shifts.\n")

        print(f"✅ EXECUTIVE_SUMMARY.txt generated successfully.")
