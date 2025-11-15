"use client";

export default function AnalyticsPage() {
  // Mock data for demonstration
  const stats = {
    totalPanels: 1247,
    processedToday: 23,
    avgRecoveryRate: 87.5,
    costSavings: 38.2,
  };

  const materialBreakdown = [
    { material: "Glass", percentage: 72, color: "bg-blue-500" },
    { material: "Aluminum", percentage: 12, color: "bg-emerald-500" },
    { material: "Silicon", percentage: 8, color: "bg-purple-500" },
    { material: "Other", percentage: 8, color: "bg-slate-500" },
  ];

  const recentClassifications = [
    { type: "Monocrystalline", count: 45, trend: "+12%" },
    { type: "Polycrystalline", count: 38, trend: "+8%" },
    { type: "Thin-Film", count: 22, trend: "-3%" },
  ];

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100 dark:from-slate-900 dark:to-slate-800">
      <div className="container mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-4xl md:text-5xl font-bold text-slate-900 dark:text-slate-100 mb-2">
            Analytics Dashboard
          </h1>
          <p className="text-slate-600 dark:text-slate-400 text-lg">
            Comprehensive insights into your solar panel recycling operations
          </p>
        </div>

        {/* Stats Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
          <div className="bg-white dark:bg-slate-800 rounded-xl shadow-lg p-6">
            <div className="flex items-center justify-between mb-2">
              <h3 className="text-sm font-medium text-slate-600 dark:text-slate-400">
                Total Panels Processed
              </h3>
              <svg
                className="w-5 h-5 text-emerald-500"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"
                />
              </svg>
            </div>
            <div className="text-3xl font-bold text-slate-900 dark:text-slate-100">
              {stats.totalPanels.toLocaleString()}
            </div>
            <p className="text-sm text-emerald-600 dark:text-emerald-400 mt-1">+15% this month</p>
          </div>

          <div className="bg-white dark:bg-slate-800 rounded-xl shadow-lg p-6">
            <div className="flex items-center justify-between mb-2">
              <h3 className="text-sm font-medium text-slate-600 dark:text-slate-400">
                Processed Today
              </h3>
              <svg
                className="w-5 h-5 text-blue-500"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z"
                />
              </svg>
            </div>
            <div className="text-3xl font-bold text-slate-900 dark:text-slate-100">
              {stats.processedToday}
            </div>
            <p className="text-sm text-blue-600 dark:text-blue-400 mt-1">On track for target</p>
          </div>

          <div className="bg-white dark:bg-slate-800 rounded-xl shadow-lg p-6">
            <div className="flex items-center justify-between mb-2">
              <h3 className="text-sm font-medium text-slate-600 dark:text-slate-400">
                Avg. Recovery Rate
              </h3>
              <svg
                className="w-5 h-5 text-purple-500"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"
                />
              </svg>
            </div>
            <div className="text-3xl font-bold text-slate-900 dark:text-slate-100">
              {stats.avgRecoveryRate}%
            </div>
            <p className="text-sm text-purple-600 dark:text-purple-400 mt-1">+2.3% improvement</p>
          </div>

          <div className="bg-white dark:bg-slate-800 rounded-xl shadow-lg p-6">
            <div className="flex items-center justify-between mb-2">
              <h3 className="text-sm font-medium text-slate-600 dark:text-slate-400">
                Cost Savings
              </h3>
              <svg
                className="w-5 h-5 text-yellow-500"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M12 8c-1.657 0-3 .895-3 2s1.343 2 3 2 3 .895 3 2-1.343 2-3 2m0-8c1.11 0 2.08.402 2.599 1M12 8V7m0 1v8m0 0v1m0-1c-1.11 0-2.08-.402-2.599-1M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
                />
              </svg>
            </div>
            <div className="text-3xl font-bold text-slate-900 dark:text-slate-100">
              {stats.costSavings}%
            </div>
            <p className="text-sm text-yellow-600 dark:text-yellow-400 mt-1">vs. traditional methods</p>
          </div>
        </div>

        {/* Charts Section */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-8">
          {/* Material Composition */}
          <div className="bg-white dark:bg-slate-800 rounded-xl shadow-lg p-6">
            <h2 className="text-xl font-bold text-slate-900 dark:text-slate-100 mb-6">
              Material Composition Breakdown
            </h2>
            <div className="space-y-4">
              {materialBreakdown.map((item) => (
                <div key={item.material}>
                  <div className="flex justify-between mb-2">
                    <span className="text-sm font-medium text-slate-700 dark:text-slate-300">
                      {item.material}
                    </span>
                    <span className="text-sm text-slate-600 dark:text-slate-400">
                      {item.percentage}%
                    </span>
                  </div>
                  <div className="w-full bg-slate-200 dark:bg-slate-700 rounded-full h-3">
                    <div
                      className={`${item.color} h-3 rounded-full transition-all duration-500`}
                      style={{ width: `${item.percentage}%` }}
                    ></div>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Panel Types */}
          <div className="bg-white dark:bg-slate-800 rounded-xl shadow-lg p-6">
            <h2 className="text-xl font-bold text-slate-900 dark:text-slate-100 mb-6">
              Classification by Type
            </h2>
            <div className="space-y-4">
              {recentClassifications.map((item) => (
                <div
                  key={item.type}
                  className="flex items-center justify-between p-4 bg-slate-50 dark:bg-slate-900 rounded-lg"
                >
                  <div>
                    <div className="font-semibold text-slate-900 dark:text-slate-100">
                      {item.type}
                    </div>
                    <div className="text-sm text-slate-600 dark:text-slate-400">
                      {item.count} panels
                    </div>
                  </div>
                  <div
                    className={`text-sm font-semibold ${
                      item.trend.startsWith("+")
                        ? "text-emerald-600 dark:text-emerald-400"
                        : "text-red-600 dark:text-red-400"
                    }`}
                  >
                    {item.trend}
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Additional Analytics Placeholder */}
        <div className="bg-white dark:bg-slate-800 rounded-xl shadow-lg p-6">
          <h2 className="text-xl font-bold text-slate-900 dark:text-slate-100 mb-4">
            Advanced Analytics
          </h2>
          <div className="text-center py-12 text-slate-500 dark:text-slate-400">
            <svg
              className="mx-auto h-16 w-16 mb-4"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"
              />
            </svg>
            <p className="text-lg mb-2">Heat Maps, Sankey Diagrams & More</p>
            <p className="text-sm">
              Connect your data source to view detailed visualizations and insights
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}

