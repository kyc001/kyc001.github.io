import { randomBytes, webcrypto } from "node:crypto";

const DEFAULT_ITERATIONS = 150000;
const subtle = (webcrypto as unknown as Crypto).subtle;

export type EncryptedContentPayload = {
	version: 1;
	algorithm: "AES-GCM";
	kdf: "PBKDF2-SHA-256";
	iterations: number;
	salt: string;
	iv: string;
	ciphertext: string;
};

const encoder = new TextEncoder();

function toBase64(value: ArrayBuffer | Uint8Array): string {
	const bytes = value instanceof Uint8Array ? value : new Uint8Array(value);
	return Buffer.from(bytes).toString("base64");
}

async function deriveEncryptionKey(
	password: string,
	salt: Uint8Array,
	iterations: number,
): Promise<CryptoKey> {
	const baseKey = await subtle.importKey(
		"raw",
		encoder.encode(password),
		"PBKDF2",
		false,
		["deriveKey"],
	);

	return subtle.deriveKey(
		{
			name: "PBKDF2",
			hash: "SHA-256",
			salt,
			iterations,
		},
		baseKey,
		{ name: "AES-GCM", length: 256 },
		false,
		["encrypt"],
	);
}

export async function encryptHtmlForPassword(
	html: string,
	password: string,
	iterations = DEFAULT_ITERATIONS,
): Promise<EncryptedContentPayload> {
	if (!password) {
		throw new Error("Protected content requires a non-empty password.");
	}

	const salt = new Uint8Array(randomBytes(16));
	const iv = new Uint8Array(randomBytes(12));
	const key = await deriveEncryptionKey(password, salt, iterations);
	const encrypted = await subtle.encrypt(
		{
			name: "AES-GCM",
			iv,
		},
		key,
		encoder.encode(html),
	);

	return {
		version: 1,
		algorithm: "AES-GCM",
		kdf: "PBKDF2-SHA-256",
		iterations,
		salt: toBase64(salt),
		iv: toBase64(iv),
		ciphertext: toBase64(encrypted),
	};
}
