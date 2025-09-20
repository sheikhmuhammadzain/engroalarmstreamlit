import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Set style for better-looking plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class AlarmDataAnalyzer:
    """Comprehensive Alarm Data Analysis and Visualization"""
    
    def __init__(self, file_path):
        """Initialize with cleaned CSV file"""
        self.df = pd.read_csv(file_path)
        self.prepare_data()
        
    def prepare_data(self):
        """Prepare data for analysis"""
        # Convert Event Time to datetime
        self.df['Event Time'] = pd.to_datetime(self.df['Event Time'])
        
        # Extract time components
        self.df['Hour'] = self.df['Event Time'].dt.hour
        self.df['Minute'] = self.df['Event Time'].dt.minute
        self.df['Date'] = self.df['Event Time'].dt.date
        
        # Clean Action column - handle NaN values
        self.df['Action'] = self.df['Action'].fillna('NO ACTION')
        
        # Create alarm categories based on conditions
        alarm_conditions = ['ALARM', 'PVHIHI', 'PVHI', 'PVLO', 'PVLOW', 'PVLOLOW', 
                          'HIHI', 'HI', 'LO', 'LOLO', 'FAIL']
        self.df['Is_Alarm'] = self.df['Condition'].str.upper().str.contains(
            '|'.join(alarm_conditions), na=False
        )
        
        print("Data preparation complete!")
        print(f"Total events: {len(self.df)}")
        print(f"Time range: {self.df['Event Time'].min()} to {self.df['Event Time'].max()}")
        
    def create_overview_dashboard(self):
        """Create a comprehensive overview dashboard"""
        fig = plt.figure(figsize=(20, 12))
        fig.suptitle('Alarm System Overview Dashboard', fontsize=20, fontweight='bold')
        
        # 1. Events over time
        ax1 = plt.subplot(3, 3, 1)
        events_per_minute = self.df.groupby(self.df['Event Time'].dt.floor('min')).size()
        ax1.plot(events_per_minute.index, events_per_minute.values, linewidth=1)
        ax1.set_title('Events Over Time (Per Minute)')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Number of Events')
        ax1.tick_params(axis='x', rotation=45)
        
        # 2. Top 10 Event Sources
        ax2 = plt.subplot(3, 3, 2)
        top_sources = self.df['Source'].value_counts().head(10)
        bars = ax2.barh(range(len(top_sources)), top_sources.values)
        ax2.set_yticks(range(len(top_sources)))
        ax2.set_yticklabels(top_sources.index)
        ax2.set_title('Top 10 Event Sources')
        ax2.set_xlabel('Count')
        # Add value labels on bars
        for i, (bar, value) in enumerate(zip(bars, top_sources.values)):
            ax2.text(value, bar.get_y() + bar.get_height()/2, f'{value}', 
                    ha='left', va='center', fontsize=8)
        
        # 3. Condition Types Distribution
        ax3 = plt.subplot(3, 3, 3)
        condition_counts = self.df['Condition'].value_counts().head(10)
        wedges, texts, autotexts = ax3.pie(condition_counts.values, 
                                            labels=condition_counts.index,
                                            autopct='%1.1f%%',
                                            startangle=90)
        ax3.set_title('Top 10 Condition Types')
        # Make percentage text smaller
        for autotext in autotexts:
            autotext.set_fontsize(8)
        
        # 4. Hourly Distribution
        ax4 = plt.subplot(3, 3, 4)
        hourly_dist = self.df['Hour'].value_counts().sort_index()
        ax4.bar(hourly_dist.index, hourly_dist.values, color='skyblue', edgecolor='navy')
        ax4.set_title('Events by Hour of Day')
        ax4.set_xlabel('Hour')
        ax4.set_ylabel('Number of Events')
        ax4.set_xticks(range(0, 24, 2))
        
        # 5. Top Locations
        ax5 = plt.subplot(3, 3, 5)
        top_locations = self.df['Location Tag'].value_counts().head(10)
        bars = ax5.bar(range(len(top_locations)), top_locations.values, color='coral')
        ax5.set_xticks(range(len(top_locations)))
        ax5.set_xticklabels(top_locations.index, rotation=45, ha='right')
        ax5.set_title('Top 10 Location Tags')
        ax5.set_ylabel('Count')
        # Add value labels on bars
        for bar, value in zip(bars, top_locations.values):
            ax5.text(bar.get_x() + bar.get_width()/2, value, f'{value}', 
                    ha='center', va='bottom', fontsize=8)
        
        # 6. Action Types
        ax6 = plt.subplot(3, 3, 6)
        action_counts = self.df['Action'].value_counts()
        colors = ['green' if 'ACK' in str(action) else 'red' for action in action_counts.index]
        bars = ax6.bar(range(len(action_counts)), action_counts.values, color=colors)
        ax6.set_xticks(range(len(action_counts)))
        ax6.set_xticklabels(action_counts.index, rotation=45, ha='right')
        ax6.set_title('Action Types Distribution')
        ax6.set_ylabel('Count')
        for bar, value in zip(bars, action_counts.values):
            ax6.text(bar.get_x() + bar.get_width()/2, value, f'{value}', 
                    ha='center', va='bottom', fontsize=8)
        
        # 7. Value Distribution (where available)
        ax7 = plt.subplot(3, 3, 7)
        value_data = self.df['Value'].dropna()
        if len(value_data) > 0:
            ax7.hist(value_data, bins=50, color='lightgreen', edgecolor='darkgreen')
            ax7.set_title(f'Value Distribution (n={len(value_data)})')
            ax7.set_xlabel('Value')
            ax7.set_ylabel('Frequency')
            # Add statistics
            ax7.axvline(value_data.mean(), color='red', linestyle='--', 
                       label=f'Mean: {value_data.mean():.2f}')
            ax7.axvline(value_data.median(), color='blue', linestyle='--', 
                       label=f'Median: {value_data.median():.2f}')
            ax7.legend()
        
        # 8. Units Distribution
        ax8 = plt.subplot(3, 3, 8)
        units_counts = self.df['Units'].value_counts().head(10)
        # Remove empty or NaN units
        units_counts = units_counts[units_counts.index != '']
        if len(units_counts) > 0:
            wedges, texts, autotexts = ax8.pie(units_counts.values, 
                                               labels=units_counts.index,
                                               autopct='%1.1f%%',
                                               startangle=90)
            ax8.set_title('Measurement Units Distribution')
            for autotext in autotexts:
                autotext.set_fontsize(8)
        
        # 9. Alarms vs Changes
        ax9 = plt.subplot(3, 3, 9)
        alarm_vs_change = self.df['Is_Alarm'].value_counts()
        labels = ['Normal Events', 'Alarms']
        colors = ['lightblue', 'salmon']
        wedges, texts, autotexts = ax9.pie(alarm_vs_change.values, 
                                           labels=labels,
                                           colors=colors,
                                           autopct='%1.1f%%',
                                           explode=(0, 0.1),
                                           startangle=90)
        ax9.set_title('Alarms vs Normal Events')
        
        plt.tight_layout()
        return fig
    
    def analyze_critical_events(self):
        """Analyze critical events and alarms"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Critical Events Analysis', fontsize=16, fontweight='bold')
        
        # Filter for alarm conditions
        alarm_df = self.df[self.df['Is_Alarm']]
        
        # 1. Alarm frequency over time
        ax1 = axes[0, 0]
        if len(alarm_df) > 0:
            alarms_per_5min = alarm_df.groupby(
                alarm_df['Event Time'].dt.floor('5min')
            ).size()
            ax1.plot(alarms_per_5min.index, alarms_per_5min.values, 
                    color='red', linewidth=2, marker='o', markersize=3)
            ax1.set_title('Alarm Frequency (5-minute intervals)')
            ax1.set_xlabel('Time')
            ax1.set_ylabel('Number of Alarms')
            ax1.grid(True, alpha=0.3)
            ax1.tick_params(axis='x', rotation=45)
        
        # 2. Most critical sources (with alarms)
        ax2 = axes[0, 1]
        if len(alarm_df) > 0:
            critical_sources = alarm_df['Source'].value_counts().head(10)
            bars = ax2.barh(range(len(critical_sources)), critical_sources.values, 
                           color='darkred')
            ax2.set_yticks(range(len(critical_sources)))
            ax2.set_yticklabels(critical_sources.index)
            ax2.set_title('Top 10 Sources with Alarms')
            ax2.set_xlabel('Alarm Count')
            for i, value in enumerate(critical_sources.values):
                ax2.text(value, i, f' {value}', va='center')
        
        # 3. Alarm types breakdown
        ax3 = axes[1, 0]
        alarm_conditions = alarm_df['Condition'].value_counts().head(10)
        if len(alarm_conditions) > 0:
            colors_list = plt.cm.Reds(np.linspace(0.3, 0.9, len(alarm_conditions)))
            bars = ax3.bar(range(len(alarm_conditions)), alarm_conditions.values, 
                          color=colors_list)
            ax3.set_xticks(range(len(alarm_conditions)))
            ax3.set_xticklabels(alarm_conditions.index, rotation=45, ha='right')
            ax3.set_title('Alarm Condition Types')
            ax3.set_ylabel('Count')
            for bar, value in zip(bars, alarm_conditions.values):
                ax3.text(bar.get_x() + bar.get_width()/2, value, f'{value}', 
                        ha='center', va='bottom')
        
        # 4. Acknowledgment status for alarms
        ax4 = axes[1, 1]
        if len(alarm_df) > 0:
            ack_status = alarm_df['Action'].apply(
                lambda x: 'Acknowledged' if 'ACK' in str(x) else 'Not Acknowledged'
            ).value_counts()
            colors = ['green', 'orange']
            wedges, texts, autotexts = ax4.pie(ack_status.values, 
                                               labels=ack_status.index,
                                               colors=colors,
                                               autopct='%1.1f%%',
                                               explode=(0.05, 0.05),
                                               startangle=90)
            ax4.set_title('Alarm Acknowledgment Status')
            
            # Add text box with statistics
            textstr = f'Total Alarms: {len(alarm_df)}\n'
            if 'Acknowledged' in ack_status.index:
                textstr += f"Acknowledged: {ack_status['Acknowledged']}\n"
            if 'Not Acknowledged' in ack_status.index:
                textstr += f"Pending: {ack_status['Not Acknowledged']}"
            ax4.text(0.95, 0.95, textstr, transform=ax4.transAxes, 
                    fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        return fig
    
    def analyze_process_variables(self):
        """Analyze process variables with values"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('Process Variables Analysis', fontsize=16, fontweight='bold')
        
        # Filter data with values
        value_df = self.df[self.df['Value'].notna()].copy()
        
        if len(value_df) > 0:
            # 1. Values over time
            ax1 = axes[0, 0]
            ax1.scatter(value_df['Event Time'], value_df['Value'], 
                       alpha=0.5, s=10, c=value_df['Value'], cmap='viridis')
            ax1.set_title('Process Values Over Time')
            ax1.set_xlabel('Time')
            ax1.set_ylabel('Value')
            ax1.tick_params(axis='x', rotation=45)
            
            # 2. Value distribution by source
            ax2 = axes[0, 1]
            top_sources_with_values = value_df['Source'].value_counts().head(5).index
            for source in top_sources_with_values:
                source_data = value_df[value_df['Source'] == source]['Value']
                ax2.hist(source_data, alpha=0.5, label=source, bins=20)
            ax2.set_title('Value Distribution by Top Sources')
            ax2.set_xlabel('Value')
            ax2.set_ylabel('Frequency')
            ax2.legend(fontsize=8)
            
            # 3. Box plot by units
            ax3 = axes[0, 2]
            units_with_values = value_df[value_df['Units'] != '']['Units'].value_counts().head(5).index
            data_for_box = []
            labels_for_box = []
            for unit in units_with_values:
                unit_data = value_df[value_df['Units'] == unit]['Value']
                if len(unit_data) > 0:
                    data_for_box.append(unit_data)
                    labels_for_box.append(f'{unit}\n(n={len(unit_data)})')
            
            if data_for_box:
                bp = ax3.boxplot(data_for_box, labels=labels_for_box)
                ax3.set_title('Value Ranges by Unit Type')
                ax3.set_ylabel('Value')
                ax3.grid(True, alpha=0.3)
            
            # 4. Moving average of values
            ax4 = axes[1, 0]
            value_df_sorted = value_df.sort_values('Event Time')
            if len(value_df_sorted) > 10:
                value_df_sorted['MA_10'] = value_df_sorted['Value'].rolling(window=10, 
                                                                            min_periods=1).mean()
                ax4.plot(value_df_sorted['Event Time'], value_df_sorted['Value'], 
                        alpha=0.3, label='Raw Values', linewidth=0.5)
                ax4.plot(value_df_sorted['Event Time'], value_df_sorted['MA_10'], 
                        color='red', linewidth=2, label='10-point Moving Avg')
                ax4.set_title('Values with Moving Average')
                ax4.set_xlabel('Time')
                ax4.set_ylabel('Value')
                ax4.legend()
                ax4.tick_params(axis='x', rotation=45)
            
            # 5. Statistics by location
            ax5 = axes[1, 1]
            location_stats = value_df.groupby('Location Tag')['Value'].agg(['mean', 'std', 'count'])
            location_stats = location_stats.sort_values('count', ascending=False).head(10)
            
            x = np.arange(len(location_stats))
            width = 0.35
            
            bars1 = ax5.bar(x - width/2, location_stats['mean'], width, 
                           label='Mean', color='skyblue')
            bars2 = ax5.bar(x + width/2, location_stats['std'].fillna(0), width, 
                           label='Std Dev', color='lightcoral')
            
            ax5.set_xlabel('Location Tag')
            ax5.set_ylabel('Value')
            ax5.set_title('Value Statistics by Location')
            ax5.set_xticks(x)
            ax5.set_xticklabels(location_stats.index, rotation=45, ha='right')
            ax5.legend()
            
            # Add count labels
            for i, (bar1, bar2, count) in enumerate(zip(bars1, bars2, location_stats['count'])):
                ax5.text(i, max(bar1.get_height(), bar2.get_height()), 
                        f'n={count}', ha='center', va='bottom', fontsize=8)
            
            # 6. Outliers detection
            ax6 = axes[1, 2]
            Q1 = value_df['Value'].quantile(0.25)
            Q3 = value_df['Value'].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = value_df[(value_df['Value'] < lower_bound) | 
                               (value_df['Value'] > upper_bound)]
            normal = value_df[(value_df['Value'] >= lower_bound) & 
                             (value_df['Value'] <= upper_bound)]
            
            ax6.scatter(normal['Event Time'], normal['Value'], 
                       alpha=0.5, s=10, label=f'Normal ({len(normal)})', color='blue')
            ax6.scatter(outliers['Event Time'], outliers['Value'], 
                       alpha=0.8, s=20, label=f'Outliers ({len(outliers)})', 
                       color='red', marker='^')
            
            ax6.axhline(y=upper_bound, color='r', linestyle='--', alpha=0.5, 
                       label=f'Upper Bound ({upper_bound:.2f})')
            ax6.axhline(y=lower_bound, color='r', linestyle='--', alpha=0.5, 
                       label=f'Lower Bound ({lower_bound:.2f})')
            
            ax6.set_title('Outlier Detection (IQR Method)')
            ax6.set_xlabel('Time')
            ax6.set_ylabel('Value')
            ax6.legend(fontsize=8)
            ax6.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        return fig
    
    def generate_summary_report(self):
        """Generate a text summary report"""
        print("\n" + "="*60)
        print("ALARM DATA ANALYSIS SUMMARY REPORT")
        print("="*60)
        
        # Basic statistics
        print("\n📊 BASIC STATISTICS:")
        print(f"  • Total Events: {len(self.df):,}")
        print(f"  • Time Period: {self.df['Event Time'].min()} to {self.df['Event Time'].max()}")
        print(f"  • Duration: {(self.df['Event Time'].max() - self.df['Event Time'].min()).total_seconds()/3600:.2f} hours")
        print(f"  • Average Events/Hour: {len(self.df)/((self.df['Event Time'].max() - self.df['Event Time'].min()).total_seconds()/3600):.2f}")
        
        # Alarm statistics
        alarm_count = self.df['Is_Alarm'].sum()
        print(f"\n🚨 ALARM STATISTICS:")
        print(f"  • Total Alarms: {alarm_count:,} ({alarm_count/len(self.df)*100:.1f}%)")
        print(f"  • Normal Events: {len(self.df) - alarm_count:,} ({(len(self.df) - alarm_count)/len(self.df)*100:.1f}%)")
        
        # Acknowledgment statistics
        ack_events = self.df[self.df['Action'].str.contains('ACK', na=False)]
        print(f"\n✅ ACKNOWLEDGMENT:")
        print(f"  • Acknowledged Events: {len(ack_events):,} ({len(ack_events)/len(self.df)*100:.1f}%)")
        print(f"  • Unacknowledged: {len(self.df) - len(ack_events):,} ({(len(self.df) - len(ack_events))/len(self.df)*100:.1f}%)")
        
        # Top issues
        print(f"\n🔝 TOP 5 EVENT SOURCES:")
        for i, (source, count) in enumerate(self.df['Source'].value_counts().head(5).items(), 1):
            print(f"  {i}. {source}: {count:,} events ({count/len(self.df)*100:.1f}%)")
        
        print(f"\n⚠️ TOP 5 CONDITIONS:")
        for i, (condition, count) in enumerate(self.df['Condition'].value_counts().head(5).items(), 1):
            print(f"  {i}. {condition}: {count:,} events ({count/len(self.df)*100:.1f}%)")
        
        print(f"\n📍 TOP 5 LOCATIONS:")
        for i, (location, count) in enumerate(self.df['Location Tag'].value_counts().head(5).items(), 1):
            print(f"  {i}. {location}: {count:,} events ({count/len(self.df)*100:.1f}%)")
        
        # Value statistics
        value_data = self.df['Value'].dropna()
        if len(value_data) > 0:
            print(f"\n📈 VALUE STATISTICS:")
            print(f"  • Events with Values: {len(value_data):,} ({len(value_data)/len(self.df)*100:.1f}%)")
            print(f"  • Mean Value: {value_data.mean():.2f}")
            print(f"  • Median Value: {value_data.median():.2f}")
            print(f"  • Std Deviation: {value_data.std():.2f}")
            print(f"  • Min Value: {value_data.min():.2f}")
            print(f"  • Max Value: {value_data.max():.2f}")
        
        # Peak hour analysis
        hourly_counts = self.df['Hour'].value_counts().sort_index()
        peak_hour = hourly_counts.idxmax()
        print(f"\n⏰ TIME ANALYSIS:")
        print(f"  • Peak Hour: {peak_hour:02d}:00 ({hourly_counts[peak_hour]:,} events)")
        print(f"  • Quietest Hour: {hourly_counts.idxmin():02d}:00 ({hourly_counts.min():,} events)")
        
        print("\n" + "="*60)
    
    def save_all_charts(self, output_dir='alarm_analysis'):
        """Save all charts to files"""
        import os
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        print(f"\n💾 Saving charts to {output_dir}/...")
        
        # Generate and save each chart
        fig1 = self.create_overview_dashboard()
        fig1.savefig(f'{output_dir}/01_overview_dashboard.png', dpi=300, bbox_inches='tight')
        print("  ✓ Overview dashboard saved")
        
        fig2 = self.analyze_critical_events()
        fig2.savefig(f'{output_dir}/02_critical_events.png', dpi=300, bbox_inches='tight')
        print("  ✓ Critical events analysis saved")
        
        fig3 = self.analyze_process_variables()
        fig3.savefig(f'{output_dir}/03_process_variables.png', dpi=300, bbox_inches='tight')
        print("  ✓ Process variables analysis saved")
        
        print(f"\n✅ All charts saved to {output_dir}/")
        
        return fig1, fig2, fig3

# Main execution
if __name__ == "__main__":
    # Initialize analyzer with your cleaned file
    analyzer = AlarmDataAnalyzer('01feb_cleaned.csv')
    
    # Generate summary report
    analyzer.generate_summary_report()
    
    # Create and display charts
    print("\n📊 Generating visualizations...")
    
    # Save all charts
    fig1, fig2, fig3 = analyzer.save_all_charts()
    
    # Display charts
    plt.show()
    
    print("\n✅ Analysis complete!")