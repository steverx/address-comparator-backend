import { config } from 'dotenv';
import { Pool } from 'pg';

config(); // Load environment variables

if (!process.env.DATABASE_URL) {
    throw new Error('DATABASE_URL is not set in environment variables');
}

console.log('Connecting to database:', process.env.DATABASE_URL.split('@')[1]); // Log only host:port/db

export const pool = new Pool({
    connectionString: process.env.DATABASE_URL,
    ssl: {
        rejectUnauthorized: false
    }
});

pool.on('connect', () => {
    console.log('Connected to Railway PostgreSQL database');
});

pool.on('error', (err) => {
    console.error('Unexpected error on idle client', err);
});