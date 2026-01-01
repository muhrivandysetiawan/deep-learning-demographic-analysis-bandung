class UltimateDemographicAI:
    """
    ULTIMATE DEMOGRAPHIC AI — OPTIMIZED — 4 METRICS (MAE, MSE, RMSE, R²)
    - Focused on village-level data (Education & Occupation features).
    - Hybrid Architecture: AutoEncoder (Feature Extraction) + ANN (Regression).
    - Comprehensive evaluation and visualization for stakeholder presentation.
    """

    def __init__(self, input_dir: str = "ultimate_clean_output", output_dir: str = "ultimate_ai_optimized_output"):
        self.input_dir = input_dir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Set Professional Plotting Style
        try:
            plt.style.use('seaborn-v0_8-whitegrid')
        except:
            plt.style.use('ggplot')
            
        plt.rcParams.update({
            'font.family': 'DejaVu Sans',
            'savefig.dpi': 300,
            'figure.dpi': 150
        })

        # State Initialization
        self.df = None
        self.X, self.y = None, None
        self.X_scaled = None
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.autoencoder, self.encoder, self.ann = None, None, None
        self.history_autoencoder, self.history_ann = None, None
        self.results = {}
        self.X_train, self.X_test, self.y_train, self.y_test = [None] * 4

    def load_and_clean_data(self) -> bool:
        """Load and process village data focusing on Education and Occupation."""
        print("🧹 Loading and cleaning demographic data...")

        filepath = os.path.join(self.input_dir, "profil_kelurahan_clean.csv")
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File {filepath} not found.")

        df = pd.read_csv(filepath)
        print(f"✅ Data loaded: {len(df):,} rows")

        # Filter categories
        df_filtered = df[df['tipe_data'].isin(['jenis_pendidikan', 'jenis_pekerjaan'])].copy()

        # Pivot Data
        try:
            pivot = df_filtered.pivot_table(
                index=['kemendagri_kode_desa_kelurahan', 'tahun', 'semester'],
                columns=['tipe_data', 'kategori'],
                values='jumlah',
                aggfunc='sum',
                fill_value=0
            )
        except Exception as e:
            print(f"⚠️ Pivot failed ({e}), falling back to groupby...")
            pivot = df_filtered.groupby([
                'kemendagri_kode_desa_kelurahan', 'tahun', 'semester', 'tipe_data', 'kategori'
            ])['jumlah'].sum().unstack(['tipe_data', 'kategori']).fillna(0)

        # Flatten Column Names
        if isinstance(pivot.columns, pd.MultiIndex):
            pivot.columns = ['_'.join(str(level) for level in col).strip() for col in pivot.columns.values]

        pivot = pivot.reset_index()

        # Target lookup: Population (Mean)
        pop_lookup = df.groupby(['kemendagri_kode_desa_kelurahan', 'tahun', 'semester'])['jumlah_penduduk'].mean().reset_index()

        # Merge Features and Target
        df_merged = pivot.merge(pop_lookup, on=['kemendagri_kode_desa_kelurahan', 'tahun', 'semester'], how='inner').fillna(0)

        # Separate Features and Target
        feature_cols = [col for col in df_merged.columns if col not in [
            'kemendagri_kode_desa_kelurahan', 'tahun', 'semester', 'jumlah_penduduk'
        ]]
        
        self.df = df_merged
        self.X = df_merged[feature_cols]
        self.y = df_merged['jumlah_penduduk']

        print(f"✅ Pre-processing complete: {len(self.X):,} samples, {self.X.shape[1]} features")
        return True

    def preprocess_data(self) -> bool:
        """Scaling and Train-Test Splitting."""
        print("⚙️  Scaling and splitting data...")

        if self.X is None or self.y is None:
            raise ValueError("Data not loaded. Call load_and_clean_data() first.")

        self.X_scaled = self.scaler_X.fit_transform(self.X)
        y_scaled = self.scaler_y.fit_transform(self.y.values.reshape(-1, 1)).flatten()

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X_scaled, y_scaled, test_size=0.2, random_state=42
        )

        print(f"✅ Split complete: {len(self.X_train)} train samples, {len(self.X_test)} test samples")
        return True

    def build_autoencoder(self, encoding_dim: int = 16) -> bool:
        """Build and train the AutoEncoder for dimensionality reduction."""
        print(f"🧠 Training AutoEncoder (Target Dim: {encoding_dim})...")

        input_dim = self.X_scaled.shape[1]
        input_layer = Input(shape=(input_dim,))
        
        # Encoder
        encoded = Dense(encoding_dim * 3, activation='relu')(input_layer)
        encoded = Dense(encoding_dim * 2, activation='relu')(encoded)
        encoded = Dense(encoding_dim, activation='relu', name='bottleneck')(encoded)

        # Decoder
        decoded = Dense(encoding_dim * 2, activation='relu')(encoded)
        decoded = Dense(encoding_dim * 3, activation='relu')(decoded)
        decoded = Dense(input_dim, activation='linear')(decoded)

        self.autoencoder = Model(input_layer, decoded)
        self.autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        self.encoder = Model(input_layer, encoded)

        # Training
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7)
        ]

        self.history_autoencoder = self.autoencoder.fit(
            self.X_train, self.X_train,
            epochs=200, batch_size=32, validation_split=0.2,
            callbacks=callbacks, verbose=0
        )

        train_loss = self.autoencoder.evaluate(self.X_train, self.X_train, verbose=0)
        test_loss = self.autoencoder.evaluate(self.X_test, self.X_test, verbose=0)

        self.results['autoencoder'] = {
            'train_loss': train_loss, 'test_loss': test_loss,
            'compression_ratio': input_dim / encoding_dim
        }

        print(f"✅ AutoEncoder Ready: Train Loss = {train_loss:.6f}, Test Loss = {test_loss:.6f}")
        return True

    def build_ann(self) -> bool:
        """Train ANN regressor on top of encoded features."""
        print("🤖 Training Deep ANN Regressor...")
        
        X_train_enc = self.encoder.predict(self.X_train, verbose=0)
        X_test_enc = self.encoder.predict(self.X_test, verbose=0)

        self.ann = Sequential([
            Input(shape=(X_train_enc.shape[1],)),
            Dense(256), LeakyReLU(negative_slope=0.01),
            BatchNormalization(), Dropout(0.2),
            Dense(128), LeakyReLU(negative_slope=0.01),
            Dropout(0.2),
            Dense(64), LeakyReLU(negative_slope=0.01),
            Dense(1, activation='linear')
        ])

        self.ann.compile(optimizer=Adam(0.001), loss='mse', metrics=['mae'])

        callbacks = [
            EarlyStopping(patience=25, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=8, min_lr=1e-6)
        ]

        self.history_ann = self.ann.fit(
            X_train_enc, self.y_train,
            epochs=400, batch_size=16, validation_split=0.15,
            callbacks=callbacks, verbose=0
        )

        # Evaluation (4 Metrics)
        y_pred_scaled = self.ann.predict(X_test_enc, verbose=0)
        y_pred = self.scaler_y.inverse_transform(y_pred_scaled).flatten()
        y_true = self.scaler_y.inverse_transform(self.y_test.reshape(-1, 1)).flatten()

        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)

        self.results['ann'] = {
            'mae': mae, 'mse': mse, 'rmse': rmse, 'r2': r2,
            'y_test_actual': y_true, 'y_pred': y_pred
        }

        print(f"✅ ANN Optimization Results:")
        print(f"   📊 R² Score = {r2:.4f}")
        print(f"   🎯 MAE      = {mae:,.0f}")
        print(f"   📈 MSE      = {mse:,.0f}")
        print(f"   ⚡ RMSE     = {rmse:,.0f}")
        return True

    def generate_visualizations(self) -> bool:
        """Produce professional English charts for GitHub/Presentation."""
        print("📊 Generating visualizations...")

        # 1. AutoEncoder Loss
        plt.figure(figsize=(10, 6))
        plt.plot(self.history_autoencoder.history['loss'], label='Training Loss')
        plt.plot(self.history_autoencoder.history['val_loss'], label='Validation Loss')
        plt.title('AutoEncoder Reconstruction Loss', fontsize=16, fontweight='bold')
        plt.xlabel('Epochs')
        plt.ylabel('Loss (MSE)')
        plt.legend()
        plt.savefig(os.path.join(self.output_dir, "autoencoder_loss.png"))
        plt.close()

        # 2. ANN Progress
        plt.figure(figsize=(10, 6))
        plt.plot(self.history_ann.history['mae'], label='Train MAE', color='#e67e22')
        plt.plot(self.history_ann.history['val_mae'], label='Val MAE', color='#d35400')
        plt.title('ANN Training Progress (MAE)', fontsize=16, fontweight='bold')
        plt.xlabel('Epochs')
        plt.ylabel('MAE (Scaled)')
        plt.legend()
        plt.savefig(os.path.join(self.output_dir, "ann_training_progress.png"))
        plt.close()

        # 3. Final Comparison & Metrics Table
        if 'ann' in self.results:
            plt.figure(figsize=(14, 10))
            
            # Scatter Plot
            plt.subplot(2, 2, 1)
            plt.scatter(self.results['ann']['y_test_actual'], self.results['ann']['y_pred'], alpha=0.7, color='#1f77b4')
            lims = [min(plt.xlim()[0], plt.ylim()[0]), max(plt.xlim()[1], plt.ylim()[1])]
            plt.plot(lims, lims, 'r--', alpha=0.75, zorder=3)
            plt.title('Actual vs. Predicted Population', fontweight='bold')
            plt.xlabel('Actual Population')
            plt.ylabel('Predicted Population')

            # Metrics Table
            plt.subplot(2, 2, 2)
            metrics_df = pd.DataFrame({
                'Metric': ['MAE', 'MSE', 'RMSE', 'R²'],
                'Value': [f"{self.results['ann']['mae']:,.0f}", f"{self.results['ann']['mse']:,.0f}",
                          f"{self.results['ann']['rmse']:,.0f}", f"{self.results['ann']['r2']:.4f}"]
            })
            table = plt.table(cellText=metrics_df.values, colLabels=metrics_df.columns, loc='center', bbox=[0, 0, 1, 1])
            table.set_fontsize(12)
            plt.title('📊 Model Performance Summary', fontweight='bold', pad=20)
            plt.axis('off')

            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, "model_performance_summary.png"))
            plt.close()

        print(f"✅ All charts saved to {self.output_dir}")
        return True

    def generate_report(self) -> bool:
        """Create executive summary and CSV outputs."""
        ae, ann = self.results.get('autoencoder', {}), self.results.get('ann', {})
        avg_pop = self.y.mean()

        report = f"""
ULTIMATE DEMOGRAPHIC AI PERFORMANCE REPORT
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
==================================================================

📊 AUTOENCODER RECONSTRUCTION:
- Training Loss: {ae.get('train_loss', 0):.6f}
- Testing Loss:  {ae.get('test_loss', 0):.6f}
- Compression Ratio: {ae.get('compression_ratio', 0):.1f}x

🤖 ANN REGRESSION PERFORMANCE:
- R² Score: {ann.get('r2', 0):.4f} (Variance Explained)
- MAE:      {ann.get('mae', 0):,.0f} ({(ann.get('mae', 0)/avg_pop)*100:.1f}% error relative to avg)
- RMSE:     {ann.get('rmse', 0):,.0f}
- Sample Size (Test): {len(ann.get('y_test_actual', [])):,}

✅ INTERPRETATION:
- R² > 0.9: Exceptional fit.
- MAE < 5% of mean: High precision for operational planning.

📁 Outputs saved in: {os.path.abspath(self.output_dir)}
"""
        with open(os.path.join(self.output_dir, "AI_PERFORMANCE_REPORT.txt"), 'w') as f:
            f.write(report)
        
        # Save Predictions CSV
        if 'ann' in self.results:
            pd.DataFrame({
                'Actual': self.results['ann']['y_test_actual'],
                'Predicted': self.results['ann']['y_pred'],
                'Residual': self.results['ann']['y_test_actual'] - self.results['ann']['y_pred']
            }).to_csv(os.path.join(self.output_dir, "predictions_summary.csv"), index=False)
            
        return True

    def run_pipeline(self):
        """Execute the full AI workflow."""
        print("🚀 Starting Ultimate Demographic AI Pipeline...")
        try:
            self.load_and_clean_data()
            self.preprocess_data()
            self.build_autoencoder()
            self.build_ann()
            self.generate_visualizations()
            self.generate_report()
            print("\n🎉 PIPELINE SUCCESSFUL — ALL ASSETS GENERATED!")
            return True
        except Exception as e:
            print(f"❌ CRITICAL ERROR: {e}")
            return False
