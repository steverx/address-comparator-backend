import { Pool } from 'pg';
import { config } from 'dotenv';

config();

export const pool = new Pool({
    connectionString: process.env.DATABASE_URL,
    ssl: {
        rejectUnauthorized: false
    }
});

export async function findAddressMatches(address: string, threshold: number) {
    try {
        const query = `
            SELECT raw_address, member_id, lic, similarity(raw_address, $1) as match_score
            FROM addresses
            WHERE similarity(raw_address, $1) > $2
            ORDER BY match_score DESC
        `;
        const result = await pool.query(query, [address, threshold / 100]);
        return result.rows;
    } catch (error) {
        console.error('Database query error:', error);
        throw new Error('Failed to search addresses');
    }
}