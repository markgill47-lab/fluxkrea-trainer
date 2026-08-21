import { render } from "preact";
import { App } from "./app";
import "./styles/fonts.css";
import "./styles/tokens.css";
import "./styles/base.css";
import "./styles/gallery.css";
import "./styles/review.css";
import "./styles/monitor.css";

const host = document.getElementById("app");
if (!host) throw new Error("no #app element");
render(<App />, host);
