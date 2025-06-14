import { ThemeProvider } from "@/components/theme-provider";

import { CompressorApp } from "@/pages/CompressorApp";

function App() {
  return (
    <ThemeProvider defaultTheme="dark" storageKey="vite-ui-theme">
      <CompressorApp />
    </ThemeProvider>
  );
}
export default App;
