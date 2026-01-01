class FinalDemographicCleaner:
    """
    Final Demographic Cleaner
    --------------------------
    A streamlined utility to deduplicate and export demographic datasets 
    into 4 specific CSV files optimized for ANN (Artificial Neural Network) processing.
    """

    def __init__(self, dataframes: Dict[str, pd.DataFrame], output_dir: str = "final_clean_output"):
        self.dataframes = dataframes
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        print(f"📁 Output directory initialized: {output_dir}")

    def clean_and_export(self) -> bool:
        """Core process: Deduplicate and export the 4 essential data files."""
        print("🧹 Starting duplication cleaning and export process...")

        # === 1. KELURAHAN PROFILE — ALL DIMENSIONS ===
        df_kelurahan = self._build_kelurahan_profile()
        if not df_kelurahan.empty:
            df_kelurahan.to_csv(os.path.join(self.output_dir, "profil_kelurahan_clean.csv"), index=False)
            print(f"✅ Exported profil_kelurahan_clean.csv: {len(df_kelurahan):,} rows")

        # === 2. KECAMATAN PROFILE — MARITAL STATUS + AGE GROUP + KTP ===
        df_kecamatan = self._build_kecamatan_profile()
        if not df_kecamatan.empty:
            df_kecamatan.to_csv(os.path.join(self.output_dir, "profil_kecamatan_clean.csv"), index=False)
            print(f"✅ Exported profil_kecamatan_clean.csv: {len(df_kecamatan):,} rows")

        # === 3. CITY TRENDS ===
        try:
            df_tren = self.dataframes['jumlah_penduduk_kota_bandung'][[
                'tahun', 'semester', 'jumlah_penduduk'
            ]].drop_duplicates().reset_index(drop=True)
            df_tren.to_csv(os.path.join(self.output_dir, "tren_kota_clean.csv"), index=False)
            print(f"✅ Exported tren_kota_clean.csv: {len(df_tren)} rows")
        except KeyError as e:
            print(f"❌ Failed to export city trends: {e}")

        # === 4. REGIONAL LOOKUP ===
        try:
            df_lookup = self.dataframes['jumlah_penduduk_kota_bandung_berdasarkan_kelurahan'][[
                'kemendagri_kode_kecamatan',
                'kemendagri_nama_kecamatan',
                'kemendagri_kode_desa_kelurahan',
                'kemendagri_nama_desa_kelurahan'
            ]].drop_duplicates().reset_index(drop=True)
            df_lookup.to_csv(os.path.join(self.output_dir, "lookup_wilayah_clean.csv"), index=False)
            print(f"✅ Exported lookup_wilayah_clean.csv: {len(df_lookup)} rows")
        except KeyError as e:
            print(f"❌ Failed to export lookup table: {e}")

        print(f"\n🎉 SUCCESS — All 4 clean files have been generated!")
        return True

    def _build_kelurahan_profile(self) -> pd.DataFrame:
        """Constructs a unified Kelurahan demographic profile."""
        try:
            # Base Kelurahan data
            df_kel = self.dataframes['jumlah_penduduk_kota_bandung_berdasarkan_kelurahan'][[
                'kemendagri_kode_desa_kelurahan',
                'kemendagri_nama_desa_kelurahan',
                'kemendagri_kode_kecamatan',
                'kemendagri_nama_kecamatan',
                'jumlah_penduduk',
                'tahun',
                'semester'
            ]].drop_duplicates().reset_index(drop=True)
        except Exception as e:
            print(f"❌ Failed to load base Kelurahan data: {e}")
            return pd.DataFrame()

        key_cols = ['kemendagri_kode_desa_kelurahan', 'tahun', 'semester']
        all_records = []

        # Demographic Dimensions
        dimensions = [
            ('gender', 'jumlah_penduduk_kota_bandung_berdasarkan_jenis_kelamin', 'jenis_kelamin'),
            ('religion', 'jumlah_penduduk_kota_bandung_berdasarkan_agama', 'agama'),
            ('occupation', 'jumlah_penduduk_kota_bandung_berdasarkan_jenis_pekerjaan', 'jenis_pekerjaan'),
            ('education', 'jumlah_penduduk_kota_bandung_berdasarkan_jenis_pendidikan', 'jenis_pendidikan'),
            ('blood_type', 'jumlah_penduduk_kota_bandung_berdasarkan_golongan_darah', 'tipe_goldar')
        ]

        for dim_name, source_name, col_name in dimensions:
            if source_name in self.dataframes:
                try:
                    df_dim = self.dataframes[source_name][[
                        'kemendagri_kode_desa_kelurahan', 'tahun', 'semester', col_name, 'jumlah_penduduk'
                    ]].drop_duplicates().reset_index(drop=True)

                    # Normalize column names
                    df_dim = df_dim.rename(columns={'jumlah_penduduk': 'count_value'})
                    df_dim['data_type'] = dim_name
                    df_dim['category'] = df_dim[col_name]

                    # Merge with base data
                    merged = df_kel.merge(df_dim, on=key_cols, how='inner')

                    # Select final columns
                    cols_needed = [
                        'kemendagri_kode_desa_kelurahan', 'kemendagri_nama_desa_kelurahan',
                        'kemendagri_kode_kecamatan', 'kemendagri_nama_kecamatan',
                        'tahun', 'semester', 'jumlah_penduduk', 'data_type', 'category', 'count_value'
                    ]

                    if all(col in merged.columns for col in cols_needed):
                        all_records.append(merged[cols_needed])

                except Exception as e:
                    print(f"⚠️ Error processing dimension '{dim_name}': {e}")
                    continue

        if not all_records:
            return pd.DataFrame()

        return pd.concat(all_records, ignore_index=True)

    def _build_kecamatan_profile(self) -> pd.DataFrame:
        """Constructs a unified Kecamatan profile with KTP coverage."""
        try:
            df_kec = self.dataframes['jumlah_penduduk_kota_bandung_berdasarkan_kecamatan'][[
                'kemendagri_kode_kecamatan',
                'kemendagri_nama_kecamatan',
                'jumlah_penduduk',
                'tahun',
                'semester'
            ]].drop_duplicates().reset_index(drop=True)
        except Exception as e:
            print(f"❌ Failed to load base Kecamatan data: {e}")
            return pd.DataFrame()

        key_cols = ['kemendagri_kode_kecamatan', 'tahun', 'semester']
        all_records = []

        # Kecamatan Dimensions
        dimensions = [
            ('marital_status', 'jumlah_penduduk_kota_bandung_berdasarkan_status_kawin', 'status_kawin'),
            ('age_group', 'jumlah_penduduk_kota_bandung_berdasarkan_kelompok_umur', 'kelompok_umur')
        ]

        for dim_name, source_name, col_name in dimensions:
            if source_name in self.dataframes:
                try:
                    df_dim = self.dataframes[source_name][[
                        'kemendagri_kode_kecamatan', 'tahun', 'semester', col_name, 'jumlah_penduduk'
                    ]].drop_duplicates().reset_index(drop=True)

                    df_dim = df_dim.rename(columns={'jumlah_penduduk': 'count_value'})
                    df_dim['data_type'] = dim_name
                    df_dim['category'] = df_dim[col_name]

                    merged = df_kec.merge(df_dim, on=key_cols, how='inner')

                    cols_needed = [
                        'kemendagri_kode_kecamatan', 'kemendagri_nama_kecamatan',
                        'tahun', 'semester', 'jumlah_penduduk', 'data_type', 'category', 'count_value'
                    ]

                    if all(col in merged.columns for col in cols_needed):
                        all_records.append(merged[cols_needed])

                except Exception as e:
                    print(f"⚠️ Error processing dimension '{dim_name}': {e}")
                    continue

        if not all_records:
            return pd.DataFrame()

        df_combined = pd.concat(all_records, ignore_index=True)

        # Integrate Identity Card (KTP) Data
        try:
            df_main = self.dataframes
            if 'jumlah_penduduk_wajib_ktp_di_kota_bandung' in df_main and \
               'jumlah_cakupan_kepemilikan_e_ktp_di_kota_bandung' in df_main:
                
                df_wajib = df_main['jumlah_penduduk_wajib_ktp_di_kota_bandung'][[
                    'kemendagri_kode_kecamatan', 'tahun', 'jumlah_wajib_ktp'
                ]].drop_duplicates().reset_index(drop=True)

                df_e_ktp = df_main['jumlah_cakupan_kepemilikan_e_ktp_di_kota_bandung'][[
                    'kemendagri_kode_kecamatan', 'tahun', 'jumlah_penduduk'
                ]].rename(columns={'jumlah_penduduk': 'total_e_ktp_owners'}).drop_duplicates().reset_index(drop=True)

                # Merge KTP metrics
                df_ktp = df_wajib.merge(df_e_ktp, on=['kemendagri_kode_kecamatan', 'tahun'], how='left')
                df_ktp['e_ktp_coverage_pct'] = (df_ktp['total_e_ktp_owners'] / df_ktp['jumlah_wajib_ktp'] * 100).round(2)

                # Append KTP metrics to combined profile
                df_combined = df_combined.merge(
                    df_ktp[['kemendagri_kode_kecamatan', 'tahun', 'jumlah_wajib_ktp', 'total_e_ktp_owners', 'e_ktp_coverage_pct']],
                    on=['kemendagri_kode_kecamatan', 'tahun'],
                    how='left'
                )
        except Exception as e:
            print(f"⚠️ Failed to integrate KTP data: {e}")

        return df_combined
