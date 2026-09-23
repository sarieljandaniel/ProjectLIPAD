import { useTelemetry } from "../hooks/useTelemetry";
import { WelcomeHero } from "../components/home/WelcomeHero";
import { TelemetryMetrics } from "../components/home/TelemetryMetrics";
import { TelemetryConsole } from "../components/home/TelemetryConsole";
import { Card } from "../components/ui/Card";

export function HomePage() {
  const { packets, connected, latestDistance } = useTelemetry();

  return (
    <div className="grid grid-cols-1 gap-0 lg:grid-cols-12">
      <section className="lg:col-span-8 lg:border-r lg:border-ink">
        <WelcomeHero />
        <div className="space-y-0">
          <TelemetryMetrics connected={connected} latestDistance={latestDistance} packets={packets} />
          <div className="mt-6">
            <TelemetryConsole packets={packets} connected={connected} />
          </div>
        </div>
      </section>
      <aside className="mt-6 border-t-4 border-ink pt-6 lg:col-span-4 lg:mt-0 lg:border-l-0 lg:border-t-0 lg:pt-0 lg:pl-6">
        <Card inverted className="h-full">
          <div className="border-b border-newsprint p-4">
            <p className="font-sans text-[10px] uppercase tracking-widest text-rule">Field guide</p>
            <h3 className="mt-2 font-serif text-xl">Connect your Pi hotspot</h3>
          </div>
          <ol className="space-y-0 p-0">
            {[
              "Enable hotspot on Raspberry Pi 4B (typically 10.42.0.1).",
              "Join the Pi network from this ground station.",
              "Ensure the Pi broadcasts UDP telemetry to port 50007.",
              "Watch live packets appear in the console at left.",
            ].map((step, i) => (
              <li key={i} className="border-b border-rule/30 px-4 py-4 last:border-0">
                <span className="font-mono text-accent">0{i + 1}</span>
                <p className="mt-1 font-sans text-sm leading-relaxed">{step}</p>
              </li>
            ))}
          </ol>
        </Card>
      </aside>
    </div>
  );
}
