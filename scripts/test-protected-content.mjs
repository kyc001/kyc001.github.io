import assert from "node:assert/strict";
import { webcrypto } from "node:crypto";
import { encryptHtmlForPassword } from "../src/utils/protected-content.ts";

const decoder = new TextDecoder();
const encoder = new TextEncoder();

function fromBase64(value) {
	return Uint8Array.from(Buffer.from(value, "base64"));
}

async function deriveKey(password, salt, iterations) {
	const baseKey = await webcrypto.subtle.importKey(
		"raw",
		encoder.encode(password),
		"PBKDF2",
		false,
		["deriveKey"],
	);

	return webcrypto.subtle.deriveKey(
		{
			name: "PBKDF2",
			hash: "SHA-256",
			salt,
			iterations,
		},
		baseKey,
		{ name: "AES-GCM", length: 256 },
		false,
		["decrypt"],
	);
}

async function decryptPayload(payload, password) {
	const key = await deriveKey(
		password,
		fromBase64(payload.salt),
		payload.iterations,
	);
	const decrypted = await webcrypto.subtle.decrypt(
		{
			name: "AES-GCM",
			iv: fromBase64(payload.iv),
		},
		key,
		fromBase64(payload.ciphertext),
	);

	return decoder.decode(decrypted);
}

const html = "<h2>Private title</h2><p>Only visible after unlock.</p>";
const password = "correct horse battery staple";
const payload = await encryptHtmlForPassword(html, password);

assert.equal(payload.version, 1);
assert.equal(payload.algorithm, "AES-GCM");
assert.equal(payload.kdf, "PBKDF2-SHA-256");
assert.equal(typeof payload.salt, "string");
assert.equal(typeof payload.iv, "string");
assert.equal(typeof payload.ciphertext, "string");
assert.ok(payload.iterations >= 100000);
assert.ok(!payload.ciphertext.includes("Private title"));

const decrypted = await decryptPayload(payload, password);
assert.equal(decrypted, html);

await assert.rejects(
	() => decryptPayload(payload, "wrong password"),
	/Error|OperationError|DOMException/,
);
