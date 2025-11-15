export default function AboutPage() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100 dark:from-slate-900 dark:to-slate-800">
      <div className="container mx-auto px-4 sm:px-6 lg:px-8 py-16">
        {/* Header */}
        <div className="text-center mb-16">
          <h1 className="text-4xl md:text-5xl font-bold text-slate-900 dark:text-slate-100 mb-4">
            About Solar<span className="text-emerald-600 dark:text-emerald-400">Vortex</span>
          </h1>
          <p className="text-xl text-slate-600 dark:text-slate-400 max-w-3xl mx-auto">
            Pioneering the future of sustainable solar panel waste management through AI and
            circular economy principles
          </p>
        </div>

        {/* Mission Section */}
        <section className="mb-16">
          <div className="bg-white dark:bg-slate-800 rounded-xl shadow-lg p-8 md:p-12">
            <h2 className="text-3xl font-bold text-slate-900 dark:text-slate-100 mb-6">
              Our Mission
            </h2>
            <p className="text-lg text-slate-600 dark:text-slate-400 leading-relaxed mb-4">
              With solar capacity projected to reach <strong>1700 GW by 2050</strong>, millions of
              tonnes of end-of-life panels will accumulate without efficient recycling methods.
              SolarVortex addresses this critical challenge by leveraging cutting-edge AI technology
              to create a sustainable circular economy for solar panel waste.
            </p>
            <p className="text-lg text-slate-600 dark:text-slate-400 leading-relaxed">
              Our platform combines computer vision, predictive analytics, and logistics optimization
              to enable automated classification, efficient material recovery, and intelligent waste
              management—reducing costs, increasing recovery rates, and preventing toxic materials
              from harming communities and ecosystems.
            </p>
          </div>
        </section>

        {/* Problem Statement */}
        <section className="mb-16">
          <div className="grid md:grid-cols-2 gap-8">
            <div className="bg-white dark:bg-slate-800 rounded-xl shadow-lg p-8">
              <h3 className="text-2xl font-bold text-slate-900 dark:text-slate-100 mb-4">
                The Challenge
              </h3>
              <ul className="space-y-3 text-slate-600 dark:text-slate-400">
                <li className="flex items-start">
                  <svg
                    className="w-6 h-6 text-red-500 mr-2 flex-shrink-0 mt-0.5"
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"
                    />
                  </svg>
                  <span>
                    Millions of tonnes of solar panel waste expected by 2050
                  </span>
                </li>
                <li className="flex items-start">
                  <svg
                    className="w-6 h-6 text-red-500 mr-2 flex-shrink-0 mt-0.5"
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"
                    />
                  </svg>
                  <span>
                    Toxic materials (Pb, Cd, Te) pose health and environmental risks
                  </span>
                </li>
                <li className="flex items-start">
                  <svg
                    className="w-6 h-6 text-red-500 mr-2 flex-shrink-0 mt-0.5"
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"
                    />
                  </svg>
                  <span>
                    Inefficient manual classification and logistics processes
                  </span>
                </li>
                <li className="flex items-start">
                  <svg
                    className="w-6 h-6 text-red-500 mr-2 flex-shrink-0 mt-0.5"
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"
                    />
                  </svg>
                  <span>
                    Low material recovery rates and high processing costs
                  </span>
                </li>
              </ul>
            </div>

            <div className="bg-white dark:bg-slate-800 rounded-xl shadow-lg p-8">
              <h3 className="text-2xl font-bold text-slate-900 dark:text-slate-100 mb-4">
                Our Solution
              </h3>
              <ul className="space-y-3 text-slate-600 dark:text-slate-400">
                <li className="flex items-start">
                  <svg
                    className="w-6 h-6 text-emerald-500 mr-2 flex-shrink-0 mt-0.5"
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"
                    />
                  </svg>
                  <span>
                    AI-powered automated classification and material identification
                  </span>
                </li>
                <li className="flex items-start">
                  <svg
                    className="w-6 h-6 text-emerald-500 mr-2 flex-shrink-0 mt-0.5"
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"
                    />
                  </svg>
                  <span>
                    Optimized logistics reducing costs by 30-40%
                  </span>
                </li>
                <li className="flex items-start">
                  <svg
                    className="w-6 h-6 text-emerald-500 mr-2 flex-shrink-0 mt-0.5"
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"
                    />
                  </svg>
                  <span>
                    Up to 95% material recovery rates
                  </span>
                </li>
                <li className="flex items-start">
                  <svg
                    className="w-6 h-6 text-emerald-500 mr-2 flex-shrink-0 mt-0.5"
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"
                    />
                  </svg>
                  <span>
                    100% landfill diversion and comprehensive analytics
                  </span>
                </li>
              </ul>
            </div>
          </div>
        </section>

        {/* Impact Section */}
        <section className="mb-16">
          <div className="bg-gradient-to-br from-emerald-600 to-emerald-700 rounded-xl shadow-lg p-8 md:p-12 text-white">
            <h2 className="text-3xl font-bold mb-8 text-center">Our Impact</h2>
            <div className="grid md:grid-cols-3 gap-8">
              <div className="text-center">
                <div className="text-4xl font-bold mb-2">Community Health</div>
                <p className="text-emerald-100">
                  Prevents hazardous exposure to toxic panel materials, protecting workers and
                  communities
                </p>
              </div>
              <div className="text-center">
                <div className="text-4xl font-bold mb-2">Economic Benefits</div>
                <p className="text-emerald-100">
                  30-40% logistics cost reduction and 15-25% higher material recovery, creating new
                  revenue streams
                </p>
              </div>
              <div className="text-center">
                <div className="text-4xl font-bold mb-2">Environmental</div>
                <p className="text-emerald-100">
                  75% lower carbon footprint and 100% landfill diversion for collected panels
                </p>
              </div>
            </div>
          </div>
        </section>

        {/* Team Section */}
        <section className="mb-16">
          <h2 className="text-3xl font-bold text-slate-900 dark:text-slate-100 mb-8 text-center">
            Our Team
          </h2>
          <div className="grid md:grid-cols-2 gap-8 max-w-4xl mx-auto">
            <div className="bg-white dark:bg-slate-800 rounded-xl shadow-lg p-8 text-center">
              <div className="w-24 h-24 bg-emerald-100 dark:bg-emerald-900/30 rounded-full flex items-center justify-center mx-auto mb-4">
                <svg
                  className="w-12 h-12 text-emerald-600 dark:text-emerald-400"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z"
                  />
                </svg>
              </div>
              <h3 className="text-xl font-semibold text-slate-900 dark:text-slate-100 mb-2">
                Rajwardhan Bhandigare
              </h3>
              <p className="text-slate-600 dark:text-slate-400 mb-2">BE IT, PICT Pune</p>
              <p className="text-sm text-slate-500 dark:text-slate-500">
                Co-founder & Developer
              </p>
            </div>

            <div className="bg-white dark:bg-slate-800 rounded-xl shadow-lg p-8 text-center">
              <div className="w-24 h-24 bg-emerald-100 dark:bg-emerald-900/30 rounded-full flex items-center justify-center mx-auto mb-4">
                <svg
                  className="w-12 h-12 text-emerald-600 dark:text-emerald-400"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z"
                  />
                </svg>
              </div>
              <h3 className="text-xl font-semibold text-slate-900 dark:text-slate-100 mb-2">
                Mrinmayi Bharve
              </h3>
              <p className="text-slate-600 dark:text-slate-400 mb-2">BE IT, PICT Pune</p>
              <p className="text-sm text-slate-500 dark:text-slate-500">
                Co-founder & Developer
              </p>
            </div>
          </div>
        </section>

        {/* Technology Stack */}
        <section>
          <div className="bg-white dark:bg-slate-800 rounded-xl shadow-lg p-8 md:p-12">
            <h2 className="text-3xl font-bold text-slate-900 dark:text-slate-100 mb-8 text-center">
              Technology Stack
            </h2>
            <div className="grid md:grid-cols-3 gap-8">
              <div>
                <h3 className="text-xl font-semibold text-slate-900 dark:text-slate-100 mb-4">
                  Frontend
                </h3>
                <ul className="space-y-2 text-slate-600 dark:text-slate-400">
                  <li>• Next.js & React</li>
                  <li>• TypeScript</li>
                  <li>• Tailwind CSS</li>
                  <li>• D3.js</li>
                  <li>• Mapbox/Google Maps</li>
                </ul>
              </div>
              <div>
                <h3 className="text-xl font-semibold text-slate-900 dark:text-slate-100 mb-4">
                  Backend
                </h3>
                <ul className="space-y-2 text-slate-600 dark:text-slate-400">
                  <li>• Node.js & Express</li>
                  <li>• PostgreSQL with PostGIS</li>
                  <li>• OAuth 2.0</li>
                  <li>• AWS S3 / Cloudinary</li>
                </ul>
              </div>
              <div>
                <h3 className="text-xl font-semibold text-slate-900 dark:text-slate-100 mb-4">
                  AI/ML
                </h3>
                <ul className="space-y-2 text-slate-600 dark:text-slate-400">
                  <li>• TensorFlow / PyTorch</li>
                  <li>• Computer Vision Models</li>
                  <li>• Time-series Forecasting</li>
                  <li>• Route Optimization</li>
                </ul>
              </div>
            </div>
          </div>
        </section>
      </div>
    </div>
  );
}

