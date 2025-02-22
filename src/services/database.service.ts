import { pool } from '../config/database.config';

export async function findAddressMatches(address: string, threshold: number = 0.8): Promise<any[]> {
    const query = `
        SELECT 
            raw_address,
            normalized_address,
            components,
            metadata
        FROM addresses;
    `;

    const result = await pool.query(query);
    return result.rows;
}