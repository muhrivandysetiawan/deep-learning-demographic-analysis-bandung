class DuplicationCleanerMerger:
    """
    Handles data deduplication and safe merging for regional population datasets.
    
    Features:
    - Multi-level duplicate detection and removal.
    - Key uniqueness validation before merging.
    - Profile building for Kelurahan and Kecamatan levels.
    - Comprehensive cleaning report and metadata export.
    """

    def __init__(self, dataframes: Dict[str, pd.DataFrame], output_dir: str = "cleaned_output"):
        self.dataframes = dataframes
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.clean_report: Dict[str, Any] = {}

    def clean_and_merge(self) -> bool:
        """Main execution pipeline: clean sources -> merge profiles -> export."""
        print(f"--- Starting Duplication Cleaning & Safe Merger ---")
        start_time = datetime.now()

        # 1. Deduplicate all source dataframes
        print("\n[1/4] Scrubbing duplicates from all data sources...")
        self._clean_all_sources()

        # 2. Build and export Kelurahan Profile
        print("\n[2/4] Building Clean Kelurahan Profiles...")
        df_kelurahan = self._build_clean_kelurahan_profile()
        if not df_kelurahan.empty:
            self._export_to_csv(df_kelurahan, "profil_kelurahan_clean.csv")
            print(f"✔ profil_kelurahan_clean.csv: {len(df_kelurahan):,} rows")

        # 3. Build and export Kecamatan Profile
        print("\n[3/4] Building Clean Kecamatan Profiles...")
        df_kecamatan = self._build_clean_kecamatan_profile()
        if not df_kecamatan.empty:
            self._export_to_csv(df_kecamatan, "profil_kecamatan_clean.csv")
            print(f"✔ profil_kecamatan_clean.csv: {len(df_kecamatan):,} rows")

        # 4. Handle City Trends and Lookup Tables
        print("\n[4/4] Processing city-wide trends and lookups...")
        self._process_trends_and_lookups()

        # Finalize reports
        duration = datetime.now() - start_time
        self.clean_report['total_duration_sec'] = duration.total_seconds()
        self._export_clean_report()

        print(f"\n🎉 PROCESS COMPLETE — All data cleaned and merged!")
        print(f"📁 Output Directory: {self.output_dir}")
        return True

    def _clean_all_sources(self) -> None:
        """Iterates through specific sources to remove duplicate records."""
        sources_to_clean = [
            'jumlah_penduduk_kota_bandung_berdasarkan_jenis_kelamin',
            'jumlah_penduduk_kota_bandung_berdasarkan_agama',
            'jumlah_penduduk_kota_bandung_berdasarkan_jenis_pekerjaan',
            'jumlah_penduduk_kota_bandung_berdasarkan_jenis_pendidikan',
            'jumlah_penduduk_kota_bandung_berdasarkan_golongan_darah',
            'jumlah_penduduk_kota_bandung_berdasarkan_status_kawin',
            'jumlah_penduduk_kota_bandung_berdasarkan_kelompok_umur',
            'jumlah_penduduk_wajib_ktp_di_kota_bandung',
            'jumlah_cakupan_kepemilikan_e_ktp_di_kota_bandung'
        ]

        population_col_map = {
            'jumlah_penduduk_wajib_ktp_di_kota_bandung': 'jumlah_wajib_ktp',
            'jumlah_cakupan_kepemilikan_e_ktp_di_kota_bandung': 'jumlah_penduduk'
        }

        for source_name in sources_to_clean:
            if source_name in self.dataframes:
                df = self.dataframes[source_name]
                original_len = len(df)
                key_cols = self._get_key_columns(source_name)
                pop_col = population_col_map.get(source_name, 'jumlah_penduduk')

                if key_cols and pop_col in df.columns:
                    subset_cols = key_cols + [pop_col]
                    self.dataframes[source_name] = df.drop_duplicates(
                        subset=subset_cols, keep='first'
                    ).reset_index(drop=True)

                    cleaned_len = len(self.dataframes[source_name])
                    removed = original_len - cleaned_len

                    if removed > 0:
                        print(f"🧹 {source_name}: {original_len:,} -> {cleaned_len:,} (Removed {removed:,} duplicates)")
                    else:
                        print(f"✅ {source_name}: Clean (No duplicates found)")

                    self.clean_report[source_name] = {
                        'original_rows': original_len,
                        'cleaned_rows': cleaned_len,
                        'duplicates_removed': removed
                    }
                else:
                    print(f"⚠️ Warning: Skipping {source_name}. Key columns or '{pop_col}' missing.")

    def _get_key_columns(self, source_name: str) -> Optional[List[str]]:
        """Determines the join/unique keys based on source name patterns."""
        if 'kelurahan' in source_name or any(x in source_name for x in ['jenis_kelamin', 'agama', 'pekerjaan', 'pendidikan', 'golongan_darah']):
            return ['kemendagri_kode_desa_kelurahan', 'tahun', 'semester']
        elif 'kecamatan' in source_name or any(x in source_name for x in ['status_kawin', 'kelompok_umur']):
            return ['kemendagri_kode_kecamatan', 'tahun', 'semester']
        elif 'wajib_ktp' in source_name or 'e_ktp' in source_name:
            return ['kemendagri_kode_kecamatan', 'tahun']
        return None

    def _build_clean_kelurahan_profile(self) -> pd.DataFrame:
        """Merges demographic dimensions into a unified Kelurahan profile."""
        try:
            df_kel = self.dataframes['jumlah_penduduk_kota_bandung_berdasarkan_kelurahan'][[
                'kemendagri_kode_desa_kelurahan', 'kemendagri_nama_desa_kelurahan',
                'kemendagri_kode_kecamatan', 'kemendagri_nama_kecamatan',
                'jumlah_penduduk', 'tahun', 'semester'
            ]].copy()
        except KeyError:
            print("⚠️ Source 'kelurahan_profile' not found. Skipping.")
            return pd.DataFrame()

        key_cols = ['kemendagri_kode_desa_kelurahan', 'tahun', 'semester']
        all_dimensions = []
        dimension_sources = [
            ('gender', 'jumlah_penduduk_kota_bandung_berdasarkan_jenis_kelamin', 'jenis_kelamin'),
            ('religion', 'jumlah_penduduk_kota_bandung_berdasarkan_agama', 'agama'),
            ('occupation', 'jumlah_penduduk_kota_bandung_berdasarkan_jenis_pekerjaan', 'jenis_pekerjaan'),
            ('education', 'jumlah_penduduk_kota_bandung_berdasarkan_jenis_pendidikan', 'jenis_pendidikan'),
            ('blood_type', 'jumlah_penduduk_kota_bandung_berdasarkan_golongan_darah', 'tipe_goldar')
        ]

        for dim_name, source_name, col_name in dimension_sources:
            if source_name in self.dataframes:
                df_dim = self.dataframes[source_name][[
                    'kemendagri_kode_desa_kelurahan', 'tahun', 'semester', col_name, 'jumlah_penduduk'
                ]].rename(columns={'jumlah_penduduk': 'value_count'})
                
                df_dim['data_type'] = dim_name
                df_dim['category'] = df_dim[col_name]

                merged = df_kel.merge(df_dim, on=key_cols, how='inner')
                cols_needed = [
                    'kemendagri_kode_desa_kelurahan', 'kemendagri_nama_desa_kelurahan',
                    'kemendagri_kode_kecamatan', 'kemendagri_nama_kecamatan',
                    'tahun', 'semester', 'jumlah_penduduk', 'data_type', 'category', 'value_count'
                ]
                if all(col in merged.columns for col in cols_needed):
                    all_dimensions.append(merged[cols_needed])

        return pd.concat(all_dimensions, ignore_index=True) if all_dimensions else pd.DataFrame()

    def _build_clean_kecamatan_profile(self) -> pd.DataFrame:
        """Merges demographic dimensions and KTP coverage into a Kecamatan profile."""
        try:
            df_kec = self.dataframes['jumlah_penduduk_kota_bandung_berdasarkan_kecamatan'][[
                'kemendagri_kode_kecamatan', 'kemendagri_nama_kecamatan',
                'jumlah_penduduk', 'tahun', 'semester'
            ]].copy()
        except KeyError:
            print("⚠️ Source 'kecamatan_profile' not found. Skipping.")
            return pd.DataFrame()

        key_cols = ['kemendagri_kode_kecamatan', 'tahun', 'semester']
        all_dimensions = []
        dimension_sources = [
            ('marital_status', 'jumlah_penduduk_kota_bandung_berdasarkan_status_kawin', 'status_kawin'),
            ('age_group', 'jumlah_penduduk_kota_bandung_berdasarkan_kelompok_umur', 'kelompok_umur')
        ]

        for dim_name, source_name, col_name in dimension_sources:
            if source_name in self.dataframes:
                df_dim = self.dataframes[source_name][[
                    'kemendagri_kode_kecamatan', 'tahun', 'semester', col_name, 'jumlah_penduduk'
                ]].rename(columns={'jumlah_penduduk': 'value_count'})
                
                df_dim['data_type'] = dim_name
                df_dim['category'] = df_dim[col_name]

                merged = df_kec.merge(df_dim, on=key_cols, how='inner')
                all_dimensions.append(merged)

        if not all_dimensions: return pd.DataFrame()
        df_combined = pd.concat(all_dimensions, ignore_index=True)

        # Integrate E-KTP Coverage Data
        try:
            df_wajib = self.dataframes.get('jumlah_penduduk_wajib_ktp_di_kota_bandung')
            df_cakupan = self.dataframes.get('jumlah_cakupan_kepemilikan_e_ktp_di_kota_bandung')

            if df_wajib is not None and df_cakupan is not None:
                df_wajib_ktp = df_wajib[['kemendagri_kode_kecamatan', 'tahun', 'jumlah_wajib_ktp']].drop_duplicates()
                df_e_ktp = df_cakupan[['kemendagri_kode_kecamatan', 'tahun', 'jumlah_penduduk']].rename(
                    columns={'jumlah_penduduk': 'jumlah_pemilik_e_ktp'}
                ).drop_duplicates()

                df_ktp = df_wajib_ktp.merge(df_e_ktp, on=['kemendagri_kode_kecamatan', 'tahun'], how='left')
                df_ktp['coverage_percentage'] = (df_ktp['jumlah_pemilik_e_ktp'] / df_ktp['jumlah_wajib_ktp'] * 100).round(2)

                df_combined = df_combined.merge(
                    df_ktp[['kemendagri_kode_kecamatan', 'tahun', 'jumlah_wajib_ktp', 'jumlah_pemilik_e_ktp', 'coverage_percentage']],
                    on=['kemendagri_kode_kecamatan', 'tahun'], how='left'
                )
        except Exception as e:
            print(f"⚠️ Failed to attach KTP data: {e}")

        return df_combined

    def _process_trends_and_lookups(self) -> None:
        """Processes and exports static lookup tables and time-series trends."""
        # City-wide Trends
        src_trend = 'jumlah_penduduk_kota_bandung'
        if src_trend in self.dataframes:
            df = self.dataframes[src_trend]
            cols = ['tahun', 'semester', 'jumlah_penduduk']
            if all(c in df.columns for c in cols):
                df_tren = df[cols].drop_duplicates().reset_index(drop=True)
                self._export_to_csv(df_tren, "tren_kota_clean.csv")

        # Regional Lookup
        src_lookup = 'jumlah_penduduk_kota_bandung_berdasarkan_kelurahan'
        if src_lookup in self.dataframes:
            df = self.dataframes[src_lookup]
            cols = ['kemendagri_kode_kecamatan', 'kemendagri_nama_kecamatan', 
                    'kemendagri_kode_desa_kelurahan', 'kemendagri_nama_desa_kelurahan']
            if all(c in df.columns for c in cols):
                df_lookup = df[cols].drop_duplicates().reset_index(drop=True)
                self._export_to_csv(df_lookup, "lookup_wilayah_clean.csv")

    def _export_to_csv(self, df: pd.DataFrame, filename: str) -> None:
        filepath = os.path.join(self.output_dir, filename)
        df.to_csv(filepath, index=False, encoding='utf-8')
        print(f"💾 Saved: {filename}")

    def _export_clean_report(self) -> None:
        """Saves deduplication stats and execution metadata."""
        report_entries = [
            {'source': src, **stats} for src, stats in self.clean_report.items() 
            if isinstance(stats, dict)
        ]
        
        if report_entries:
            pd.DataFrame(report_entries).to_csv(
                os.path.join(self.output_dir, "CLEANING_REPORT.csv"), index=False
            )

        metadata = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'total_duration_sec': self.clean_report.get('total_duration_sec', 0),
            'files_processed': len(report_entries)
        }
        with open(os.path.join(self.output_dir, "METADATA.json"), 'w') as f:
            json.dump(metadata, f, indent=4)
