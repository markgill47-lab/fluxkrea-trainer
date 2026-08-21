import { render } from "preact";
import { App } from "./app";
import "./styles/tokens.css";
import "./styles/base.css";
import "./styles/review.css";

const host = document.getElementById("app");
if (!host) throw new Error("no #app element");
render(<App />, host);
